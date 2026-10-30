// RLBridge.cc — Bonn talk-group conversation process + RL bridge
//
// Conversation process (Aschenbruck IPCCC 2007 + LCN 2006):
//   • 5 independent talk groups: 4 local (one per area) + 1 global
//   • Per-group semi-Markov model: CONV_IDLE → IN_CALL → BETWEEN_CALLS → ...
//   • All random draws use a dedicated std::mt19937_64 seeded from episodeSeed_
//     combined with a "traffic" constant — guarantees byte-identical replays.
//
#include <deque>
#include "inet/networklayer/common/L3AddressTag_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "RLBridge.h"
#include <cstdlib>
#include "UserBonnTrafficApp.h"
#include "RLControlledMobility.h"
#include <map>
#include <algorithm>
#include "inet/mobility/base/MobilityBase.h"
#include "inet/networklayer/common/NetworkInterface.h"
#include "inet/networklayer/contract/IInterfaceTable.h"
#include "inet/networklayer/ipv4/IIpv4RoutingTable.h"
#include "inet/networklayer/ipv4/Ipv4Route.h"
#include "inet/common/INETMath.h"
#include "inet/common/Simsignals.h"
#include "inet/common/packet/Packet.h"
#include "inet/linklayer/common/MacAddressTag_m.h"
#include "inet/linklayer/ieee80211/mac/Ieee80211Frame_m.h"
#include "inet/linklayer/ieee802154/Ieee802154MacHeader_m.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/SignalTag_m.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IRadio.h"

#include <omnetpp/ccanvas.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

namespace inet {

using json = nlohmann::json;

// ─── Diagnostic instrumentation (count-only, no observation impact) ─────────
// Counts MAC/PHY events behind the per-user RSSI measurement (radio-level vs
// MAC-level packetSentToUpper hook). Lives entirely in this .cc file as
// file-local state keyed by drone idx.
namespace {
struct DiagCounters {
    long radioBranchHits          = 0;  // src->getName()=="radio" hits at the packetSentToUpper hook
    std::map<std::string, long>   srcNameHits;  // distinct src->getName() seen at that hook (any name)
    long powerTagYes              = 0;  // of radioBranchHits: SignalPowerInd tag present
    long powerTagNo               = 0;  // of radioBranchHits: SignalPowerInd tag absent
    long powerNan                 = 0;  // of powerTagYes: power value was NaN (never reaches resolution)
    long resolvedViaMacTag        = 0;  // (a) MacAddressInd present
    long resolvedViaHeaderFallback= 0;  // (b) Ieee80211 header transmitter-address fallback succeeded
    long resolveFailed            = 0;  // (c) neither worked
    long mappedToUser             = 0;  // of resolved: mac -> macToUserIdx_
    long mappedToDrone             = 0; // of resolved: mac -> macToDroneIdx_
    long mappedToNeither           = 0; // of resolved: mac in neither map
    long receptionEndedTotal      = 0;  // INDEPENDENT count: radio-level receptionEndedSignal,
                                         // fired unconditionally in Radio::endReception for every
                                         // completed reception (successful or corrupted) — does not
                                         // depend on src->getName()=="radio" matching, nor on
                                         // packetSentToUpper firing at all. Ground truth for "is the
                                         // PHY receiving anything".

    // ── Diagnostic: TX side, does this node's radio ever transmit ──
    long transmissionStarted      = 0;  // Radio::transmissionStartedSignal, unconditional
    long transmissionEnded        = 0;  // Radio::transmissionEndedSignal, unconditional
};
std::map<int, DiagCounters> g_diag;      // keyed by drone idx

// ── Diagnostic: does this USER ever receive a drone-sourced frame? ──
// Mirrors the drone-side radio-branch resolution logic above but on the
// user side, count-only (does not write any RSSI/observation state — users
// have no RSSI arrays in RLBridge to begin with).
struct DiagUserCounters {
    long radioBranchHits = 0;  // this user's radio-level packetSentToUpper hits
    long fromDrone        = 0;  // resolved source mac -> macIsDrone_==true
    long fromUser         = 0;  // resolved source mac -> macIsDrone_==false
    long unresolved       = 0;  // no SignalPowerInd / no resolvable source mac
    long transmissionStarted = 0;  // this user's own radio transmissionStartedSignal
    long transmissionEnded   = 0;
};
std::map<int, DiagUserCounters> g_diagUser;  // keyed by user idx
} // namespace

Define_Module(RLBridge);

RLBridge::RLBridge() = default;

RLBridge::~RLBridge()
{
    cancelAndDelete(syncEvent_);
    for (auto& p : tgProcs_) {
        cancelAndDelete(p.convIdleTimer);
        cancelAndDelete(p.callEndTimer);
        cancelAndDelete(p.callStartTimer);
        cancelAndDelete(p.packetTimer);
    }
}

// ─── initialize ───────────────────────────────────────────────────────────────

void RLBridge::initialize(int stage)
{
    if (stage == INITSTAGE_LOCAL) {
        maxClusters_      = (int)par("maxClusters").intValue();
        droneSpawnMode_   = par("droneSpawnMode").stdstringValue();
        userSpawnMode_    = par("userSpawnMode").stdstringValue();
        exploreP_         = par("exploreP").doubleValue();
        dumbbellD_        = par("dumbbellD").doubleValue();
        dumbbellJitter_   = par("dumbbellJitter").doubleValue();
        maxNeighbourSep_  = par("maxNeighbourSep").doubleValue();
        droneSpawnJitter_ = par("droneSpawnJitter").doubleValue();
        endpoint_     = par("pythonEndpoint").stringValue();
        syncInterval_ = par("syncInterval");
        numDrones_    = par("numDrones");
        numUsers_     = par("numUsers");
        maxSteps_     = par("maxSteps");
        episodeSeed_  = par("episodeSeed");
        placementMode_ = par("placementMode").stdstringValue();
        userSpread_    = par("userSpread");
        trafficMode_   = par("trafficMode").stdstringValue();

        // CUT decision, read once (see RLBridge.h cutNotForUs_). Reported in
        // MODE: because the cut is invisible downstream — with it on, the
        // event stream simply looks sparse, which is exactly what a broken
        // event pipeline also looks like. configured != executed is what this
        // line exists to catch.
        {
            const char* keepEnv = std::getenv("OMNET_KEEP_NOTFORUS");
            cutNotForUs_ = !(keepEnv && keepEnv[0] == '1');
        }
        std::printf("MODE: trafficMode=%s placementMode=%s "
                    "cutNotForUs=%d eventCap=%d\n",
                    trafficMode_.c_str(), par("placementMode").stringValue(),
                    cutNotForUs_ ? 1 : 0, MAX_EVENTS_CAP);
        if (!cutNotForUs_ && MAX_EVENTS_CAP < 1024) {
            std::printf("MODE: WARNING keeping NOT_ADDRESSED_TO_US at "
                        "eventCap=%d — raw rate is ~480/drone/step (p99 1022), "
                        "so real receptions WILL be truncated by arrival "
                        "order. Raise MAX_EVENTS_CAP to >=1024.\n",
                        MAX_EVENTS_CAP);
        }
        std::fflush(stdout);
        pCross_        = par("pCross");
        senderProb_    = par("senderProb");
        arenaWidth_   = par("arenaWidth");
        arenaHeight_  = par("arenaHeight");
        incidentZoneRadius_ = par("incidentZoneRadius");
        commandZoneRadius_  = par("commandZoneRadius");
        ccsZoneRadius_      = par("ccsZoneRadius");
        fieldPerIncidentMin_ = par("fieldPerIncidentMin");
        fieldPerIncidentMax_ = par("fieldPerIncidentMax");
        ccsStaffPerZoneMin_  = par("ccsStaffPerZoneMin");
        ccsStaffPerZoneMax_  = par("ccsStaffPerZoneMax");

        actionBuffer_.assign(numDrones_, {std::nan(""), std::nan("")});
        sendGateBuffer_.assign(numDrones_, 1.0);   // default: always send
        drones_.assign(numDrones_, nullptr);
        droneUserCounts_.assign(numDrones_, 0);
        droneBytesRx_.assign(numDrones_, 0L);
        droneBytesUp_.assign(numDrones_, 0L);
        // OM2G state-message accounting (inert on the OMNET track).
        ctrlBytesTx_.assign(numDrones_, 0L);
        ctrlBytesRx_.assign(numDrones_, 0L);
        beaconTxCount_.assign(numDrones_, 0L);
        beaconRxCount_.assign(numDrones_, 0L);
        beaconRxTime_.assign(numDrones_, std::vector<SimTime>(numDrones_, SimTime::ZERO));
        beaconRxValid_.assign(numDrones_, std::vector<double>(numDrones_, 0.0));
        userBytesRx_.assign(numUsers_, 0L);
        userBytesUp_.assign(numUsers_, 0L);
        userToDrone_.assign(numUsers_, -1);
        userIps_.assign(numUsers_, Ipv4Address::UNSPECIFIED_ADDRESS);

        sampledUserPositions_.assign(numUsers_, Coord::ZERO);
        sampledUserAreas_.assign(numUsers_, 0);
        sampledUserIncidentIds_.assign(numUsers_, 0);
        sampledUserRolesInternal_.assign(numUsers_, "field");
        trafficApps_.assign(numUsers_, nullptr);
        talkGroupHeads_.assign(5, -1);
        activeSpeakers_.assign(5, -1);

        packetReceivedSig_    = registerSignal("packetReceivedFromLower");
        packetSentToUpperSig_ = registerSignal("packetSentToUpper");
        getParentModule()->subscribe(packetReceivedSig_,    this);
        getParentModule()->subscribe(packetSentToUpperSig_, this);
        // DROP_ANY diagnostic: drone-only subscriptions below cannot see drops on
        // user[] nodes (the handler walks up looking for a module named "host").
        // Subscribing at the network level catches every drop in the topology.
        getParentModule()->subscribe(registerSignal("packetDropped"), this);

        // ── Diagnostic MAC/PHY event-count measurement (no observation impact) ──
        // Subscribe directly on each drone's wlan[0].mac submodule so only that
        // drone's own MAC stack (mac + its dcf/hcf/rx/ds children) is counted —
        // not IP-layer/AODV drops, which share the same signal names.
        packetDroppedSig_ = registerSignal("packetDropped");
        linkBrokenSig_    = registerSignal("linkBroken");
        droneEventCounts_.assign(numDrones_, EventCounts{});
        droneEvents_.assign(numDrones_, {});
        droneBusyTime_.assign(numDrones_, 0.0);
        droneLastStateChangeTime_.assign(numDrones_, SimTime::ZERO);
        droneReceptionState_.assign(numDrones_, (int)physicallayer::IRadio::RECEPTION_STATE_IDLE);
        droneLastRxTime_.assign(numDrones_, SimTime::ZERO);
        peerDroneRssiDbm_.assign(numDrones_, std::vector<double>(numDrones_, 0.0));
        peerDroneRssiValid_.assign(numDrones_, std::vector<double>(numDrones_, 0.0));
        peerDroneRssiTime_.assign(numDrones_, std::vector<SimTime>(numDrones_, SimTime::ZERO));
        userRssiDbm_.assign(numDrones_, std::vector<double>(numUsers_, 0.0));
        userRssiValid_.assign(numDrones_, std::vector<double>(numUsers_, 0.0));
        userRssiTime_.assign(numDrones_, std::vector<SimTime>(numUsers_, SimTime::ZERO));
        {
            cModule* net = getParentModule();
            for (int i = 0; i < numDrones_; ++i) {
                cModule* host = net->getSubmodule("host", i);
                cModule* wlan = host ? host->getSubmodule("wlan", 0) : nullptr;
                cModule* mac  = wlan ? wlan->getSubmodule("mac") : nullptr;
                if (mac) {
                    mac->subscribe(packetDroppedSig_, this);
                    mac->subscribe(linkBrokenSig_, this);
                }
            }
        }
        // channel_busy_fraction: subscribe directly on each drone's own radio
        // for CCA state transitions. Verified against INET 4.6
        // Radio::updateTransceiverState(), which emits
        // IRadio::receptionStateChangedSignal (source = the radio module
        // itself) only when RECEPTION_STATE_{IDLE,BUSY,RECEIVING,UNDEFINED}
        // actually changes — never periodically — so busy time must be
        // integrated across transitions, not sampled.
        {
            cModule* net = getParentModule();
            for (int i = 0; i < numDrones_; ++i) {
                cModule* host  = net->getSubmodule("host", i);
                cModule* wlan  = host  ? host->getSubmodule("wlan", 0) : nullptr;
                cModule* radio = wlan  ? wlan->getSubmodule("radio")   : nullptr;
                if (radio) {
                    radio->subscribe(physicallayer::IRadio::receptionStateChangedSignal, this);
                    // Diagnostic (see DiagCounters above): independent,
                    // name-filter-free count of completed receptions at the radio.
                    radio->subscribe(receptionEndedSignal, this);
                    // Diagnostic: does this drone's radio ever transmit
                    // anything at all (app data, AODV control, ACKs, beacons)?
                    radio->subscribe(transmissionStartedSignal, this);
                    radio->subscribe(transmissionEndedSignal, this);
                }
            }
            // Diagnostic: same TX subscription on USER radios, for
            // comparison — confirms the instrumentation itself sees transmissions
            // when they do happen (users are known to transmit MELPE traffic).
            for (int i = 0; i < numUsers_; ++i) {
                cModule* user  = net->getSubmodule("user", i);
                cModule* wlan  = user  ? user->getSubmodule("wlan", 0) : nullptr;
                cModule* radio = wlan  ? wlan->getSubmodule("radio")   : nullptr;
                if (radio) {
                    radio->subscribe(transmissionStartedSignal, this);
                    radio->subscribe(transmissionEndedSignal, this);
                }
            }
        }
        // Network-layer packetDropped: host[i].ipv4.ip (Ipv4NetworkLayer's
        // submodule named "ipv4" -> its Ipv4 submodule named "ip" — verified
        // against inet/networklayer/ipv4/Ipv4NetworkLayer.ned and
        // NetworkLayerNodeBase.ned; matches Aodv.ned's own
        // networkProtocolModule default "^.ipv4.ip"). AODV (host[i].ipv4.ip's
        // sibling "aodv" module) does not emit packetDropped at all — verified
        // against inet/routing/aodv/Aodv.cc, which has no emit() calls — so
        // there is no AODV-level subscription to add.
        {
            cModule* net = getParentModule();
            for (int i = 0; i < numDrones_; ++i) {
                cModule* host     = net->getSubmodule("host", i);
                cModule* ipv4Mod  = host ? host->getSubmodule("ipv4") : nullptr;
                cModule* ip       = ipv4Mod ? ipv4Mod->getSubmodule("ip") : nullptr;
                if (ip)
                    ip->subscribe(packetDroppedSig_, this);
            }
        }

        ctx_  = std::make_unique<zmq::context_t>(1);
        sock_ = std::make_unique<zmq::socket_t>(*ctx_, zmq::socket_type::req);
        sock_->set(zmq::sockopt::linger, 0);
        sock_->connect(endpoint_);

        // Drone-pair range lines.
        cCanvas* canvas = getParentModule()->getCanvas();
        for (int i = 0; i < numDrones_; ++i) {
            for (int j = i + 1; j < numDrones_; ++j) {
                std::string n = "dronePair_" + std::to_string(i) + "_" + std::to_string(j);
                auto* line = new cLineFigure(n.c_str());
                line->setLineStyle(cFigure::LINE_DASHED);
                line->setLineColor(cFigure::Color("grey"));
                line->setLineWidth(1.0);
                line->setZIndex(-0.5);
                line->setVisible(false);
                canvas->addFigure(line);
                dronePairLines_.push_back(line);
            }
        }

        syncEvent_ = new cMessage("rlSync");
        scheduleAt(SimTime(0), syncEvent_);
    }
    else if (stage == INITSTAGE_TRANSPORT_LAYER) {
        std::mt19937 rng;
        if (episodeSeed_ == 0)
            rng.seed(static_cast<uint32_t>(intrand(0x7fffffff)));
        else
            rng.seed(static_cast<uint32_t>(episodeSeed_));
        sampleEpisodeLayout(rng);
        sampleDroneSpawns(rng);   // AFTER users: drones are placed relative to clusters

        // Locate jammer module (optional — network may not have one).
        jammerModule_ = getParentModule()->getSubmodule("jammer");
        if (jammerModule_) {
            cModule* mob = jammerModule_->getSubmodule("mobility");
            if (mob && mob->hasPar("activate"))
                jammerActivateStep_ = (int)mob->par("activate").intValue();
            EV_INFO << "RLBridge: jammer found, activate step=" << jammerActivateStep_ << endl;
        }

        // Build the MAC-address -> is-drone registry used by the raw
        // per-event observation set (receiveSignal, EventRecord.fromDrone).
        // Done here (INITSTAGE_TRANSPORT_LAYER), not INITSTAGE_LOCAL, because
        // NetworkInterface auto-assigns real MAC addresses at
        // INITSTAGE_LINK_LAYER — reading getMacAddress() any earlier would
        // return the unspecified default.
        {
            cModule* net = getParentModule();
            for (int i = 0; i < numDrones_; ++i) {
                cModule* host = net->getSubmodule("host", i);
                if (!host) continue;
                auto* ift = check_and_cast<IInterfaceTable*>(host->getSubmodule("interfaceTable"));
                NetworkInterface* wlan = ift->findInterfaceByName("wlan0");
                if (wlan) {
                    macIsDrone_[wlan->getMacAddress()]    = true;
                    macToDroneIdx_[wlan->getMacAddress()] = i;
                }
            }
            for (int i = 0; i < numUsers_; ++i) {
                cModule* user = net->getSubmodule("user", i);
                if (!user) continue;
                auto* ift = check_and_cast<IInterfaceTable*>(user->getSubmodule("interfaceTable"));
                NetworkInterface* wlan = ift->findInterfaceByName("wlan0");
                if (wlan) {
                    macIsDrone_[wlan->getMacAddress()]   = false;
                    macToUserIdx_[wlan->getMacAddress()] = i;
                }
            }

            // Diagnostic: dump the registry
            // right after it's built, to confirm drone MACs are actually
            // registered (as opposed to e.g. wlan0 not existing yet, or
            // getMacAddress() returning the unspecified default).
            std::printf("[DIAG_MACMAP] macToDroneIdx_ size=%zu:\n", macToDroneIdx_.size());
            for (auto& kv : macToDroneIdx_)
                std::printf("[DIAG_MACMAP]   drone[%d] mac=%s\n",
                            kv.second, kv.first.str().c_str());
            std::printf("[DIAG_MACMAP] macToUserIdx_ size=%zu:\n", macToUserIdx_.size());
            for (auto& kv : macToUserIdx_)
                std::printf("[DIAG_MACMAP]   user[%d] mac=%s\n",
                            kv.second, kv.first.str().c_str());
            std::printf("[DIAG_MACMAP] macIsDrone_ size=%zu:\n", macIsDrone_.size());
            for (auto& kv : macIsDrone_)
                std::printf("[DIAG_MACMAP]   mac=%s isDrone=%d\n",
                            kv.first.str().c_str(), kv.second ? 1 : 0);
            std::fflush(stdout);
        }
    }
}

// ─── Episode layout ───────────────────────────────────────────────────────────

void RLBridge::samplePadreClusters(std::mt19937& rng)
{
    // ---- CONTROLLED-SEPARATION SPAWN -------------------------------------
    // Centres on a horizontal line, pairwise separation pinned to dumbbellD_.
    // The production path below places centres uniformly at random, which makes
    // separation/r_d a per-seed lottery and is the measured source of the
    // bimodal baseline scores (some seeds 0, some fine). Here separation is a
    // controlled variable.
    if (userSpawnMode_ == "archipelago" || userSpawnMode_ == "dumbbell") {
        const double W = arenaWidth_, H = arenaHeight_;
        const double sp = userSpread_ * (W / 2000.0);
        int nCl = (maxClusters_ > 0) ? maxClusters_ : 2;
        if (nCl < 1) nCl = 1;
        const double margin = dumbbellJitter_;      // wall margin, reused param
        const double minSep = dumbbellD_;           // enforced pairwise separation

        // Uniform-random centres, REJECTION-SAMPLED to a minimum pairwise
        // separation. Random placement alone makes separation/r_d a per-seed
        // lottery, which is the measured cause of the bimodal baselines (some
        // seeds 0, some fine). Pinning the MINIMUM keeps per-episode geometry
        // entropy -- which is what the learned message has to carry -- while
        // removing the lottery.
        std::uniform_real_distribution<double> ux(margin, W - margin),
                                               uy(margin, H - margin);
        // Resample the WHOLE set until the separation BAND holds: every pair at
        // least minSep apart, and every cluster's NEAREST neighbour no further
        // than maxNeighbourSep_. A minimum alone leaves the tail unbounded and
        // produces bridges too deep for the first relay rung to be learnable.
        std::vector<double> cx(nCl), cy(nCl);
        bool band = false;
        for (int attempt = 0; attempt < 600 && !band; ++attempt) {
            for (int c2 = 0; c2 < nCl; ++c2) {
                bool ok = false;
                for (int tries = 0; tries < 4000 && !ok; ++tries) {
                    cx[c2] = ux(rng); cy[c2] = uy(rng);
                    ok = true;
                    for (int d2 = 0; d2 < c2; ++d2) {
                        double dx2 = cx[c2] - cx[d2], dy2 = cy[c2] - cy[d2];
                        if (std::sqrt(dx2 * dx2 + dy2 * dy2) < minSep) { ok = false; break; }
                    }
                }
            }
            band = true;
            for (int c2 = 0; c2 < nCl && band; ++c2) {
                double nn = 1e18;
                for (int d2 = 0; d2 < nCl; ++d2) {
                    if (d2 == c2) continue;
                    double dx2 = cx[c2] - cx[d2], dy2 = cy[c2] - cy[d2];
                    nn = std::min(nn, std::sqrt(dx2 * dx2 + dy2 * dy2));
                }
                if (nCl > 1 && nn > maxNeighbourSep_) band = false;
            }
        }

        // Dirichlet-ish cluster sizes: the DEMAND ENTROPY. Which bridge is
        // worth building depends on these counts, they are observable only as
        // heard-user state at the drones sitting on clusters, and they reach a
        // relay ONLY through the transmitted message -- peer positions cannot
        // carry them even in principle. This is what makes the payload
        // irreplaceable rather than decorative.
        // $OM2G_DEMAND_ALPHA / $OM2G_DEMAND_FLOOR tune HOW SKEWED the load is.
        // alpha=1.0, floor=0.15 is the historical draw and remains the default,
        // so an unset environment reproduces every number measured before this.
        // Lower alpha concentrates users onto one island; the floor pulls the
        // draw back toward uniform, so BOTH must come down to make the hot
        // cluster genuinely hot. The skew is the entire point -- if every
        // island carries the same load there is nothing about the demand that
        // is worth transmitting, and no message can pay for its own airtime.
        static const double demandAlpha = std::getenv("OM2G_DEMAND_ALPHA")
            ? std::atof(std::getenv("OM2G_DEMAND_ALPHA")) : 1.0;
        static const double demandFloor = std::getenv("OM2G_DEMAND_FLOOR")
            ? std::atof(std::getenv("OM2G_DEMAND_FLOOR")) : 0.15;
        std::gamma_distribution<double> gam(demandAlpha > 0.0 ? demandAlpha : 1.0, 1.0);
        std::vector<double> gw(nCl); double gsum = 0.0;
        for (int c2 = 0; c2 < nCl; ++c2) { gw[c2] = gam(rng) + demandFloor; gsum += gw[c2]; }
        std::vector<int> sizes(nCl, 1); int assigned = 0;
        for (int c2 = 0; c2 < nCl; ++c2) {
            sizes[c2] = std::max(1, (int)std::lround(gw[c2] / gsum * numUsers_));
            assigned += sizes[c2];
        }
        while (assigned != numUsers_) {
            int diff = (numUsers_ > assigned) ? +1 : -1, idx = 0;
            if (diff > 0) { for (int c2=1;c2<nCl;++c2) if (sizes[c2]>sizes[idx]) idx=c2; }
            else          { for (int c2=0;c2<nCl;++c2) if (sizes[c2]>1){ idx=c2; break; } }
            sizes[idx] += diff; assigned += diff;
        }

        std::normal_distribution<double> g(0.0, sp);
        int u2 = 0;
        for (int c2 = 0; c2 < nCl && u2 < numUsers_; ++c2) {
            for (int k = 0; k < sizes[c2] && u2 < numUsers_; ++k, ++u2) {
                // Truncate at 2.5 sigma: the analytic cap needs the cluster
                // radius to be a HARD bound, not a tail probability.
                double dx2, dy2;
                do { dx2 = g(rng); dy2 = g(rng); }
                while (std::sqrt(dx2 * dx2 + dy2 * dy2) > 2.5 * sp);
                double x = std::min(W, std::max(0.0, cx[c2] + dx2));
                double y = std::min(H, std::max(0.0, cy[c2] + dy2));
                sampledUserPositions_[u2]     = Coord(x, y, 0.0);
                sampledUserAreas_[u2]         = c2;
                sampledUserIncidentIds_[u2]   = 0;
                sampledUserRolesInternal_[u2] = "field";
            }
        }
        std::printf("SPAWN_DIAG: archipelago nCl=%d minSep=%.1f margin=%.1f sp=%.1f "
                    "alpha=%.2f floor=%.2f sizes=", nCl, minSep, margin, sp,
                    demandAlpha, demandFloor);
        for (int c2 = 0; c2 < nCl; ++c2) std::printf("%d,", sizes[c2]);
        std::printf(" c0=(%.1f,%.1f) users=%d\n", cx[0], cy[0], numUsers_);
        std::fflush(stdout);
        return;
    }

    // PADRE Environments.py port (users only; drones placed elsewhere).
    //   cluster count: geometric(p=1/(10*alpha)), alpha=0.6, clamped [2,numUsers]
    //   cluster sizes: Dirichlet(beta*10=1.0 per cluster)*numUsers, integer-fixup
    //   centers:       uniform [0,arena]^2
    //   users:         Gaussian(center, userSpread_) per axis, clipped [0,arena]
    //   area label:    cluster index (feeds cross-cluster reward downstream)
    const double alpha = 0.6, beta = 0.1;
    const double W = arenaWidth_, H = arenaHeight_;

    // cluster count
    double scale = alpha * 10.0;                       // = 6.0
    std::geometric_distribution<int> geo(1.0 / scale); // support {0,1,...}
    int numClusters = geo(rng) + 1;                    // -> {1,2,...} like numpy
    numClusters = std::max(2, std::min(numClusters, numUsers_));
    if (maxClusters_ > 0) numClusters = std::min(numClusters, maxClusters_);   // SS8.3 cap
    std::printf("SPAWN_DIAG: clusters=%d (cap=%d) users=%d\n",
                numClusters, maxClusters_, numUsers_);
    std::fflush(stdout);

    // Dirichlet sizes via normalized gammas
    std::gamma_distribution<double> gam(beta * 10.0, 1.0);
    std::vector<double> g(numClusters);
    double gsum = 0.0;
    for (int c = 0; c < numClusters; ++c) { g[c] = gam(rng); gsum += g[c]; }
    if (gsum <= 0.0) gsum = 1.0;
    std::vector<int> sizes(numClusters, 1);
    int assigned = 0;
    for (int c = 0; c < numClusters; ++c) {
        sizes[c] = std::max(1, (int)std::lround(g[c] / gsum * numUsers_));
        assigned += sizes[c];
    }
    while (assigned != numUsers_) {
        int diff = (numUsers_ > assigned) ? +1 : -1;
        int idx = 0;
        if (diff > 0) { for (int c=1;c<numClusters;++c) if (sizes[c]>sizes[idx]) idx=c; }
        else          { for (int c=0;c<numClusters;++c) if (sizes[c]>1){ idx=c; break; } }
        sizes[idx] += diff; assigned += diff;
    }

    // centers + users
    std::uniform_real_distribution<double> ux(0.0, W), uy(0.0, H);
    // Cluster spread SCALES WITH ARENA so cluster *shape* is L-invariant and only
    // the SEPARATION between clusters grows. A flat spread means a bigger arena just
    // yields fatter clusters and the relay problem never appears. This mirrors
    // task_validation.py's `sp = spread * (L / 2000.0)` -- the selection sampler and
    // the simulator MUST produce the same layout, so the arena is chosen on the world
    // that is simulated.
    const double sp = userSpread_ * (W / 2000.0);
    std::normal_distribution<double> gx(0.0, sp), gy(0.0, sp);
    int u = 0;
    for (int c = 0; c < numClusters && u < numUsers_; ++c) {
        double cx = ux(rng), cy = uy(rng);
        for (int k = 0; k < sizes[c] && u < numUsers_; ++k, ++u) {
            double x = std::min(W, std::max(0.0, cx + gx(rng)));
            double y = std::min(H, std::max(0.0, cy + gy(rng)));
            sampledUserPositions_[u]   = Coord(x, y, 0.0);
            sampledUserAreas_[u]       = c;          // cluster id = area label
            sampledUserIncidentIds_[u] = 0;          // unused in padre mode
            sampledUserRolesInternal_[u] = "field";  // harmless default
        }
    }
    EV_INFO << "RLBridge: PADRE spawn — clusters=" << numClusters
            << " users=" << numUsers_ << " seed=" << episodeSeed_ << endl;
}

void RLBridge::sampleDroneSpawns(std::mt19937& rng)
{
    // Place drones relative to the user clusters that were JUST sampled.
    // Runs at INITSTAGE_TRANSPORT_LAYER (users exist; drones have registered).
    //
    // Purpose: the ini hardcodes 5 drone positions in a decorative cross
    // that has no relationship to where the users actually spawned this episode.
    // At L=1400 that means corner drones ~990 m apart against r_d=525 m -> the
    // backbone starts in 3 disconnected pieces (comps=3, routes=31/85) and, under
    // a random policy, never reconnects. Meanwhile a drone can spawn 600 m from
    // any user: the reward is a decode CLIFF (exactly 0 below threshold, not a
    // fade), so it sees reward 0 in every direction, has no gradient, and never
    // moves. Both are properties of the scenario layout.
    if (droneSpawnMode_ == "ini") return;              // legacy: honour initialX/initialY

    // EXPLORING STARTS. Training-only: with probability exploreP_ the drones
    // start uniformly at random rather than on the clusters, so the states
    // that contain a relay chain appear in the data from step 0. Without this
    // a policy that always starts parked on clusters may never sample the
    // configuration it is supposed to learn. Evaluation uses "cluster".
    if (droneSpawnMode_ == "explore") {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        double roll = u01(rng);

        // Cluster centroids from the users as actually placed this episode.
        std::map<int, std::pair<Coord,int>> cacc;
        for (int u = 0; u < numUsers_; ++u) {
            auto& e = cacc[sampledUserAreas_[u]];
            e.first += sampledUserPositions_[u];
            e.second += 1;
        }
        std::vector<Coord> ctr;
        for (auto& kv : cacc)
            if (kv.second.second > 0)
                ctr.push_back(kv.second.first / (double)kv.second.second);

        // ---- SCAFFOLD: start ALREADY PART-WAY across a bridge --------------
        // Formation is the hard step; holding is easy. Seeing near-formed
        // bridges from frame 0 lets the policy learn hold-and-value first.
        // Exploring-starts mix, from the environment (see patch_mixenv.py).
        // Defaults reproduce 25% scaffold / 25% hull-uniform / 50% cluster.
        const char* envSc = std::getenv("OM2G_P_SCAFFOLD");
        const char* envRn = std::getenv("OM2G_P_RANDOM");
        double pScaffold = envSc ? std::atof(envSc) : 0.25;
        double pRandom   = envRn ? std::atof(envRn) : 0.50;
        if (roll < pScaffold && ctr.size() >= 2
                && numDrones_ >= (int)ctr.size() + 2) {
            // SCAFFOLD_V2: geometry derived from the radio, not guessed.
            std::uniform_int_distribution<int> pick(0, (int)ctr.size() - 1);
            int a = pick(rng);
            // b := NEAREST cluster to a. The band constraint guarantees this is
            // within maxNeighbourSep, so the gap is always bridgeable inside
            // the drone budget; a uniform pair was often a diagonal one that
            // no 2-3 relay chain could ever close.
            int b = -1; double bestD = 1e18;
            for (int k = 0; k < (int)ctr.size(); ++k) {
                if (k == a) continue;
                double dx = ctr[k].x - ctr[a].x, dy = ctr[k].y - ctr[a].y;
                double dk = std::sqrt(dx*dx + dy*dy);
                if (dk < bestD) { bestD = dk; b = k; }
            }
            if (b < 0) b = (a + 1) % (int)ctr.size();
            // Target hop 380 m leaves 525 - 380 = 145 m of headroom for jitter.
            int nrel = (int)std::ceil(bestD / 380.0) - 1;
            if (nrel < 2) nrel = 2;
            if (nrel > 4) nrel = 4;
            // sigma 60 m: a 366 m hop at 2 sigma is 486 m, still inside 525.
            std::normal_distribution<double> jit(0.0, 60.0);
            int d = 0;
            for (int k = 1; k <= nrel && d < numDrones_; ++k, ++d) {
                double t = (double)k / (nrel + 1);
                Coord p = ctr[a] * (1.0 - t) + ctr[b] * t;
                check_and_cast<RLControlledMobility*>(drones_[d])
                    ->setSpawnPosition(
                        std::min(arenaWidth_,  std::max(0.0, p.x + jit(rng))),
                        std::min(arenaHeight_, std::max(0.0, p.y + jit(rng))));
            }
            for (int k = 0; d < numDrones_ && d < (int)drones_.size(); ++k, ++d) {
                Coord p = ctr[k % ctr.size()];
                check_and_cast<RLControlledMobility*>(drones_[d])
                    ->setSpawnPosition(p.x, p.y);
            }
            std::printf("SPAWN_DIAG: explore SCAFFOLD pair=%d-%d relays=%d D=%.0f hop=%.0f\n",
                        a, b, nrel, bestD, bestD / (nrel + 1));
            std::fflush(stdout);
            return;
        }

        bool rnd = (roll < pRandom);
        if (rnd) {
            // Uniform over the HULL of the centres + margin, not the whole
            // arena: corner spawns are wasted frames.
            double lo_x = 1e18, hi_x = -1e18, lo_y = 1e18, hi_y = -1e18;
            for (auto& p : ctr) {
                lo_x = std::min(lo_x, p.x); hi_x = std::max(hi_x, p.x);
                lo_y = std::min(lo_y, p.y); hi_y = std::max(hi_y, p.y);
            }
            if (ctr.empty()) { lo_x = lo_y = 0.0; hi_x = arenaWidth_; hi_y = arenaHeight_; }
            std::uniform_real_distribution<double>
                ux(std::max(0.0, lo_x - 300.0), std::min(arenaWidth_,  hi_x + 300.0)),
                uy(std::max(0.0, lo_y - 300.0), std::min(arenaHeight_, hi_y + 300.0));
            for (int d = 0; d < numDrones_ && d < (int)drones_.size(); ++d)
                check_and_cast<RLControlledMobility*>(drones_[d])
                    ->setSpawnPosition(ux(rng), uy(rng));
            std::printf("SPAWN_DIAG: explore RANDOM p=%.2f drones=%d\n",
                        exploreP_, numDrones_);
            std::fflush(stdout);
            return;
        }
        std::printf("SPAWN_DIAG: explore CLUSTER p=%.2f drones=%d\n",
                    exploreP_, numDrones_);
        std::fflush(stdout);
        droneSpawnMode_ = "cluster";   // fall through to the cluster branch
    }

    // "depot": all drones launch from one fixed base, far from the users.
    // NOT the same as "ini": that mode returns above without ever calling
    // setSpawnPosition(), so the coordinates never reach the mobility module
    // and the reward (which reads getCurrentPosition()) never changes. This
    // goes through the same setSpawnPosition() path as staging/cluster.
    if (droneSpawnMode_ == "depot") {
        const double dcx = par("depotX").doubleValue();
        const double dcy = par("depotY").doubleValue();
        const double dr  = par("depotRing").doubleValue();
        for (int d = 0; d < numDrones_; ++d) {
            if (!drones_[d]) continue;
            const double a = 2.0 * M_PI * d / std::max(1, numDrones_);
            const double x = std::min(arenaWidth_,  std::max(0.0, dcx + dr * std::cos(a)));
            const double y = std::min(arenaHeight_, std::max(0.0, dcy + dr * std::sin(a)));
            check_and_cast<RLControlledMobility*>(drones_[d])->setSpawnPosition(x, y);
            std::printf("SPAWN_DIAG: drone=%d mode=depot pos=(%.0f,%.0f) ring=%.0fm\n",
                        d, x, y, dr);
        }
        std::fflush(stdout);
        return;
    }

    // "uniform": drones land uniformly at random over the arena, DECORRELATED
    // from where the users are. Every centroid-based mode (staging/cluster)
    // hands the policy the answer before step 0 -- a drone spawned ON a cluster
    // already knows where the demand is, so a peer message has nothing left to
    // tell it. Decorrelating the spawn is the PRECONDITION for measuring a
    // communication lever at all: each drone then starts with a different
    // heard-set, which is exactly what gives a peer message something to add.
    //
    // Deliberately NOT "depot": at a single base every drone hears nobody and
    // all observations are identical, so a scripted argmax moves the whole
    // swarm in lockstep and BOTH arms freeze together -- the lever would read
    // zero for a reason that has nothing to do with communication.
    if (droneSpawnMode_ == "uniform") {
        const double m = std::min(200.0, 0.1 * std::min(arenaWidth_, arenaHeight_));
        std::uniform_real_distribution<double> ux(m, arenaWidth_  - m),
                                               uy(m, arenaHeight_ - m);
        for (int d = 0; d < numDrones_; ++d) {
            if (!drones_[d]) continue;
            const double x = ux(rng), y = uy(rng);
            check_and_cast<RLControlledMobility*>(drones_[d])->setSpawnPosition(x, y);
            std::printf("SPAWN_DIAG: drone=%d mode=uniform pos=(%.0f,%.0f)\n", d, x, y);
        }
        std::fflush(stdout);
        return;
    }

    // Cluster centroids, from the users as actually placed (post-clipping).
    std::map<int, std::pair<Coord,int>> acc;           // area -> (sum, count)
    for (int u = 0; u < numUsers_; ++u) {
        int a = sampledUserAreas_[u];
        auto& e = acc[a];
        e.first += sampledUserPositions_[u];
        e.second += 1;
    }
    std::vector<Coord> centroids;
    std::vector<int>   counts;
    for (auto& kv : acc) {
        if (kv.second.second <= 0) continue;
        centroids.push_back(kv.second.first / (double)kv.second.second);
        counts.push_back(kv.second.second);
    }
    if (centroids.empty()) {
        std::printf("SPAWN_DIAG: no clusters, falling back to ini positions\n");
        std::fflush(stdout);
        return;
    }

    // Which centroid does each drone get?
    std::vector<int> pick(numDrones_, 0);
    if (droneSpawnMode_ == "staging") {
        // A: everyone on the LARGEST cluster. Starts connected; other clusters
        // are over the horizon and must be found across a zero-reward valley.
        int biggest = 0;
        for (int c = 1; c < (int)counts.size(); ++c)
            if (counts[c] > counts[biggest]) biggest = c;
        std::fill(pick.begin(), pick.end(), biggest);
    }
    else if (droneSpawnMode_ == "cluster") {
        // B (default config): shuffled round-robin. Deal drones out one per
        // cluster, wrapping. Guarantees every cluster is covered (uniform random
        // assignment can put all 5 on one cluster by chance -> a dead episode and
        // an outlier that drags the median). The SHUFFLE keeps which cluster gets
        // the spare drone varying per episode.
        std::vector<int> order(centroids.size());
        for (int c = 0; c < (int)order.size(); ++c) order[c] = c;
        std::shuffle(order.begin(), order.end(), rng);
        for (int d = 0; d < numDrones_; ++d)
            pick[d] = order[d % (int)order.size()];
    }
    else {
        throw cRuntimeError("RLBridge: unknown droneSpawnMode '%s' "
                            "(expected ini|cluster|staging|depot|uniform)", droneSpawnMode_.c_str());
    }

    std::normal_distribution<double> jit(0.0, droneSpawnJitter_);
    for (int d = 0; d < numDrones_; ++d) {
        if (!drones_[d]) continue;
        Coord c = centroids[pick[d]];
        double x = std::min(arenaWidth_,  std::max(0.0, c.x + jit(rng)));
        double y = std::min(arenaHeight_, std::max(0.0, c.y + jit(rng)));
        check_and_cast<RLControlledMobility*>(drones_[d])->setSpawnPosition(x, y);
        std::printf("SPAWN_DIAG: drone=%d mode=%s cluster=%d/%d pos=(%.0f,%.0f) jitter=%.0fm\n",
                    d, droneSpawnMode_.c_str(), pick[d], (int)centroids.size(),
                    x, y, droneSpawnJitter_);
    }
    std::fflush(stdout);
}

void RLBridge::sampleEpisodeLayout(std::mt19937& rng)
{
    auto randDouble = [&](double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(rng);
    };
    auto randInt = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };

    // PADRE-style spawn: Gaussian clusters, cross-cluster reward via area=cluster.
    // Bonn code below is untouched (byte-identical when placementMode_=="bonn").
    if (placementMode_ == "padre") {
        samplePadreClusters(rng);
        // Traffic processes must still be built — the early return below used
        // to skip the build at the bottom of this function, leaving tgProcs_
        // empty (no timers -> no events -> sim ends at t=0).
        std::printf("BUILD_DIAG: trafficMode_=[%s] -> %s (padre spawn path)\n",
                    trafficMode_.c_str(),
                    (trafficMode_ == "padre") ? "buildPadreSenders" : "buildTalkGroups");
        std::fflush(stdout);
        if (trafficMode_ == "padre")
            buildPadreSenders();
        else
            buildTalkGroups();
        drawZoneCircles();
        return;
    }

    // Determine number of incidents (used for internal zone assignment).
    int maxK = (numUsers_ - 1) / (1 + fieldPerIncidentMin_ + ccsStaffPerZoneMin_);
    int maxIncidents = std::min(3, std::max(1, maxK));
    numSampledIncidents_ = randInt(1, maxIncidents);

    // Sample zone centers.
    commandZoneCenter_ = Coord(randDouble(0, arenaWidth_), randDouble(0, arenaHeight_), 0);
    patientsWaitingCenter_ = Coord(randDouble(0, arenaWidth_), randDouble(0, arenaHeight_), 0);
    incidentZoneCenters_.clear();
    ccsZoneCenters_.clear();
    for (int i = 0; i < numSampledIncidents_; ++i) {
        incidentZoneCenters_.emplace_back(randDouble(0, arenaWidth_), randDouble(0, arenaHeight_), 0);
        ccsZoneCenters_.emplace_back(randDouble(0, arenaWidth_), randDouble(0, arenaHeight_), 0);
    }

    // Sample CCS staff counts.
    std::vector<int> ccsCountPerZone(numSampledIncidents_);
    int totalCCSStaff = 0;
    for (int i = 0; i < numSampledIncidents_; ++i) {
        ccsCountPerZone[i] = randInt(ccsStaffPerZoneMin_, ccsStaffPerZoneMax_);
        totalCCSStaff += ccsCountPerZone[i];
    }
    // Clamp CCS to fit within numUsers_.
    int needed2 = 1 + numSampledIncidents_ + totalCCSStaff;
    while (totalCCSStaff > 0 && needed2 > numUsers_) {
        for (int i = numSampledIncidents_ - 1; i >= 0 && needed2 > numUsers_; --i) {
            if (ccsCountPerZone[i] > ccsStaffPerZoneMin_) {
                --ccsCountPerZone[i]; --totalCCSStaff;
                needed2 = 1 + numSampledIncidents_ + totalCCSStaff;
            }
        }
        break;
    }

    // ── Internal role assignment (used for positioning + area mapping) ────────
    // user[0]              = command-zone head
    // user[1..k]           = section heads (at incident zones)
    // user[k+1..k+CCS]     = CCS staff
    // user[remaining]      = field responders

    sampledUserRolesInternal_[0]  = "command";
    sampledUserIncidentIds_[0]    = -1;
    for (int i = 0; i < numSampledIncidents_; ++i) {
        sampledUserRolesInternal_[1 + i] = "section";
        sampledUserIncidentIds_[1 + i]   = i;
    }
    int firstCCS = 1 + numSampledIncidents_;
    int ccsIdx   = firstCCS;
    for (int i = 0; i < numSampledIncidents_; ++i) {
        for (int s = 0; s < ccsCountPerZone[i]; ++s, ++ccsIdx) {
            if (ccsIdx >= numUsers_) break;
            sampledUserRolesInternal_[ccsIdx] = "ccs";
            sampledUserIncidentIds_[ccsIdx]   = i;
        }
    }
    int actualTotalCCS = ccsIdx - firstCCS;
    int firstField = firstCCS + actualTotalCCS;
    for (int j = firstField; j < numUsers_; ++j) {
        sampledUserRolesInternal_[j] = "field";
        sampledUserIncidentIds_[j]   = (j - firstField) % numSampledIncidents_;
    }

    // ── Area assignment ───────────────────────────────────────────────────────
    // area 3 (command):          user 0
    // area 0 (incident):         section leaders + even-indexed field responders
    // area 2 (CCS):              CCS staff
    // area 1 (patients-waiting): odd-indexed field responders
    sampledUserAreas_[0] = 3;
    for (int i = 0; i < numSampledIncidents_; ++i)
        sampledUserAreas_[1 + i] = 0;
    for (int idx = firstCCS; idx < firstField; ++idx)
        sampledUserAreas_[idx] = 2;
    int fieldIdx = 0;
    for (int j = firstField; j < numUsers_; ++j, ++fieldIdx)
        sampledUserAreas_[j] = (fieldIdx % 2 == 0) ? 0 : 1;

    // ── Position sampling ────────────────────────────────────────────────────
    sampledUserPositions_[0] = sampleInDisk(rng, commandZoneCenter_, commandZoneRadius_);
    for (int i = 0; i < numSampledIncidents_; ++i)
        sampledUserPositions_[1 + i] = sampleInDisk(rng, incidentZoneCenters_[i], incidentZoneRadius_);
    ccsIdx = firstCCS;
    for (int i = 0; i < numSampledIncidents_; ++i) {
        for (int s = 0; s < ccsCountPerZone[i]; ++s, ++ccsIdx) {
            if (ccsIdx >= numUsers_) break;
            sampledUserPositions_[ccsIdx] = sampleInDisk(rng, ccsZoneCenters_[i], ccsZoneRadius_);
        }
    }
    fieldIdx = 0;
    for (int j = firstField; j < numUsers_; ++j, ++fieldIdx) {
        if (fieldIdx % 2 == 0) {
            sampledUserPositions_[j] = sampleInDisk(rng,
                incidentZoneCenters_[sampledUserIncidentIds_[j]], incidentZoneRadius_);
        } else {
            sampledUserPositions_[j] = sampleInDisk(rng, patientsWaitingCenter_, incidentZoneRadius_);
        }
    }

    EV_INFO << "RLBridge: episode layout — incidents=" << numSampledIncidents_
            << " totalCCS=" << actualTotalCCS
            << " seed=" << episodeSeed_ << endl;
    for (int i = 0; i < numUsers_; ++i)
        EV_INFO << "  user[" << i << "] area=" << sampledUserAreas_[i]
                << " pos=(" << sampledUserPositions_[i].x << ","
                << sampledUserPositions_[i].y << ")" << endl;

    std::printf("BUILD_DIAG: trafficMode_=[%s] -> %s\n", trafficMode_.c_str(),
                (trafficMode_ == "padre") ? "buildPadreSenders" : "buildTalkGroups");
    std::fflush(stdout);
    if (trafficMode_ == "padre")
        buildPadreSenders();
    else
        buildTalkGroups();
    drawZoneCircles();
}

// ─── Talk-group construction (5 groups: 4 local + 1 global) ──────────────────

void RLBridge::buildTalkGroups()
{
    talkGroups_.clear();
    tgProcs_.clear();

    // Collect members per area.
    std::vector<std::vector<int>> areaMembers(4);
    for (int u = 0; u < numUsers_; ++u)
        areaMembers[sampledUserAreas_[u]].push_back(u);
    for (int a = 0; a < 4; ++a)
        std::sort(areaMembers[a].begin(), areaMembers[a].end());

    talkGroupHeads_.assign(5, -1);

    // 4 local groups (tgId 0-3, matching area_id).
    for (int a = 0; a < 4; ++a) {
        int head = areaMembers[a].empty() ? -1 : areaMembers[a][0];
        talkGroupHeads_[a] = head;

        TalkGroup tg;
        tg.type     = "local_" + std::to_string(a);
        tg.incident = a;
        tg.head     = head;
        tg.members  = areaMembers[a];
        talkGroups_.push_back(std::move(tg));

        TalkGroupProcess p;
        p.id      = a;
        p.head    = head;
        p.members = areaMembers[a];
        std::string pfx = "tg" + std::to_string(a) + "_";
        p.convIdleTimer  = new cMessage((pfx + "convIdle").c_str());
        p.callEndTimer   = new cMessage((pfx + "callEnd").c_str());
        p.callStartTimer = new cMessage((pfx + "callStart").c_str());
        p.packetTimer    = new cMessage((pfx + "packet").c_str());
        tgProcs_.push_back(std::move(p));
    }

    // Global group (tgId 4): all users, head = lowest user in area 3 (command).
    int globalHead = talkGroupHeads_[3];
    if (globalHead < 0) globalHead = 0;  // fallback
    talkGroupHeads_[4] = globalHead;

    std::vector<int> allUsers(numUsers_);
    std::iota(allUsers.begin(), allUsers.end(), 0);

    TalkGroup gtg;
    gtg.type     = "global";
    gtg.incident = -1;
    gtg.head     = globalHead;
    gtg.members  = allUsers;
    talkGroups_.push_back(std::move(gtg));

    TalkGroupProcess gp;
    gp.id      = 4;
    gp.head    = globalHead;
    gp.members = allUsers;
    gp.convIdleTimer  = new cMessage("tg4_convIdle");
    gp.callEndTimer   = new cMessage("tg4_callEnd");
    gp.callStartTimer = new cMessage("tg4_callStart");
    gp.packetTimer    = new cMessage("tg4_packet");
    tgProcs_.push_back(std::move(gp));

    activeSpeakers_.assign(5, -1);

    EV_INFO << "RLBridge: 5 talk groups — heads=["
            << talkGroupHeads_[0] << "," << talkGroupHeads_[1] << ","
            << talkGroupHeads_[2] << "," << talkGroupHeads_[3] << ","
            << talkGroupHeads_[4] << "]" << endl;
    for (int i = 0; i < 5; ++i)
        EV_INFO << "  TG[" << i << "] head=" << tgProcs_[i].head
                << " members=" << tgProcs_[i].members.size() << endl;
}

// ─── PADRE per-sender process construction ────────────────────────────────────
// Replaces buildTalkGroups() when trafficMode_=="padre": every selected
// sender gets its own single-partner process instead of the 5 talk groups.
// The bursty call-timing engine (onConvIdleExpired/startCall/onCallStart/
// onPacketTick/onCallEnd) is unchanged and drives these processes exactly
// like it drives the Bonn talk-group ones.
void RLBridge::buildPadreSenders()
{
    talkGroups_.clear();
    tgProcs_.clear();

    // Sender/partner selection below runs during episode-layout sampling
    // (INITSTAGE_TRANSPORT_LAYER), before startConversationProcesses() seeds
    // trafficRng_ for the runtime call-timing engine. Seed it here from
    // episodeSeed_ so the selection is reproducible per episode;
    // startConversationProcesses() reseeds trafficRng_ again later, so this
    // early use does not affect runtime traffic-generation determinism.
    trafficRng_.seed(episodeSeed_ == 0
        ? static_cast<uint64_t>(intrand(0x7fffffff))
        : static_cast<uint64_t>(episodeSeed_));

    // Select senders: each user independently with probability senderProb_.
    std::vector<int> senders;
    for (int u = 0; u < numUsers_; ++u)
        if (uniformTraffic() < senderProb_) senders.push_back(u);

    // Guarantee at least one sender if none were picked (and there is
    // at least one user to pick from).
    if (senders.empty() && numUsers_ > 0)
        senders.push_back(uniformIntTraffic(0, numUsers_ - 1));

    // Pair every sender FIRST, so the demand floor below can retarget
    // conversations before any timer objects exist.
    std::vector<std::pair<int,int>> convs;
    for (int u : senders) {
        int partner = pickPadrePartner(u);
        if (partner < 0) continue;  // no valid partner for this sender — skip
        convs.emplace_back(u, partner);
    }

    // $OM2G_PAIR_FLOOR: guarantee at least one conversation on EVERY cluster
    // pair. Measured at n=30, 17% of cluster pairs carry no traffic at all, so
    // a drone that CORRECTLY bridges such a pair earns exactly nothing — the
    // reward gives no credit for a right relay in the wrong place. That is a
    // cliff, and PPO does not climb cliffs (adverse gradient across the
    // intermediate region). The floor turns it into a slope.
    //
    // It deliberately does NOT flatten demand: it steals from the BUSIEST pair,
    // one conversation at a time, so the Dirichlet concentration survives (the
    // top pair still carries ~46% of demand). That matters because knowing
    // WHERE the demand is is what the state message carries -- flatten it and
    // the value of the message disappears with it.
    static const bool pairFloor = (std::getenv("OM2G_PAIR_FLOOR") != nullptr);
    if (pairFloor && !convs.empty()) {
        int nC = 0;
        for (int u = 0; u < numUsers_; ++u)
            nC = std::max(nC, sampledUserAreas_[u] + 1);
        std::vector<std::vector<int>> byCluster(std::max(nC, 1));
        for (int u = 0; u < numUsers_; ++u)
            byCluster[sampledUserAreas_[u]].push_back(u);

        auto key = [](int a, int b) {
            return std::make_pair(std::min(a, b), std::max(a, b));
        };
        std::map<std::pair<int,int>, std::vector<int>> covered;
        for (int i = 0; i < (int)convs.size(); ++i) {
            int a = sampledUserAreas_[convs[i].first];
            int b = sampledUserAreas_[convs[i].second];
            if (a != b) covered[key(a, b)].push_back(i);
        }

        int retargeted = 0, uncovered = 0;
        for (int a = 0; a < nC; ++a) {
            for (int b = a + 1; b < nC; ++b) {
                if (covered.count(key(a, b))) continue;
                ++uncovered;
                if (byCluster[a].empty() || byCluster[b].empty()) continue;
                // Steal from the busiest pair that can spare one, so the floor
                // costs the concentration as little as possible.
                auto donor = covered.end();
                for (auto it = covered.begin(); it != covered.end(); ++it)
                    if (it->second.size() > 1 &&
                        (donor == covered.end() || it->second.size() > donor->second.size()))
                        donor = it;
                if (donor == covered.end()) break;   // nothing spare to move
                int idx = donor->second.back();
                donor->second.pop_back();
                convs[idx] = std::make_pair(
                    byCluster[a][uniformIntTraffic(0, (int)byCluster[a].size() - 1)],
                    byCluster[b][uniformIntTraffic(0, (int)byCluster[b].size() - 1)]);
                covered[key(a, b)].push_back(idx);
                ++retargeted;
            }
        }
        std::printf("PAIR_FLOOR: clusters=%d uncovered=%d retargeted=%d convs=%d\n",
                    nC, uncovered, retargeted, (int)convs.size());
        std::fflush(stdout);
    }

    for (auto& conv : convs) {
        const int u       = conv.first;
        const int partner = conv.second;

        TalkGroupProcess p;
        p.id           = (int)tgProcs_.size();
        p.current      = u;
        p.other        = partner;
        p.head         = u;
        p.members      = {u};
        p.isMesh       = false;
        p.fixedPartner = partner;

        std::string pfx = "sender" + std::to_string(u) + "_";
        p.convIdleTimer  = new cMessage((pfx + "convIdle").c_str());
        p.callEndTimer   = new cMessage((pfx + "callEnd").c_str());
        p.callStartTimer = new cMessage((pfx + "callStart").c_str());
        p.packetTimer    = new cMessage((pfx + "packet").c_str());
        tgProcs_.push_back(std::move(p));
    }

    activeSpeakers_.assign(tgProcs_.size(), -1);

    std::printf("PADRE_DIAG: senders=%d tgProcs=%d numUsers=%d senderProb=%.3f\n",
                (int)senders.size(), (int)tgProcs_.size(), numUsers_, senderProb_);
    std::fflush(stdout);

    EV_INFO << "RLBridge: PADRE traffic — " << tgProcs_.size()
            << " sender process(es) out of " << numUsers_ << " users" << endl;
    for (auto& p : tgProcs_)
        EV_INFO << "  sender=" << p.head << " partner=" << p.fixedPartner << endl;
}

// ─── Conversation-process startup ─────────────────────────────────────────────

void RLBridge::startConversationProcesses()
{
    if (!trafficParams_.paramsLoaded) {
        EV_WARN << "RLBridge: no traffic apps registered — conversation processes disabled" << endl;
        return;
    }

    // Seed traffic RNG deterministically from episode_seed + "traffic" constant.
    uint64_t trafficSeed;
    if (episodeSeed_ == 0) {
        trafficSeed = static_cast<uint64_t>(intrand(0x7fffffff));
    } else {
        // splitmix64-style mix: episode_seed xored with hash of "traffic"
        trafficSeed = static_cast<uint64_t>(episodeSeed_) * 6364136223846793005ULL
                      + 0x9e3779b97f4a7c15ULL;  // "traffic" magic constant
        trafficSeed ^= (trafficSeed >> 30);
        trafficSeed *= 0xbf58476d1ce4e5b9ULL;
        trafficSeed ^= (trafficSeed >> 27);
        trafficSeed *= 0x94d049bb133111ebULL;
        trafficSeed ^= (trafficSeed >> 31);
    }
    trafficRng_.seed(trafficSeed);

    // Print citation sanity: the ini values used by this run.
    std::printf("CITATION_SANITY: headInitiatorProb=%.3f meshProb=%.3f "
                "transHH=%.3f transMM=%.3f contProb=%.1f "
                "cht_mu=%.7f cht_sd=%.7f "
                "idle_shape=%.6f idle_rate=%.6f "
                "conv_mu=%.6f conv_sd=%.6f conv_shift=%.1f "
                "melpe_ms=%.1f\n",
                trafficParams_.headInitiatorProb, trafficParams_.meshProb,
                trafficParams_.transHeadToHead, trafficParams_.transMemberToMember,
                trafficParams_.callContinuationProb,
                trafficParams_.callCHT_meanlog, trafficParams_.callCHT_sdlog,
                trafficParams_.callIdle_shape, trafficParams_.callIdle_rate,
                trafficParams_.convIdle_meanlog, trafficParams_.convIdle_sdlog,
                trafficParams_.convIdle_shift, trafficParams_.melpeIntervalMs);
    std::fflush(stdout);

    // Schedule first conv-idle expiry for each group with >= 2 members.
    // PADRE sender processes have exactly 1 member (the sender itself) by
    // design — their partner lives in fixedPartner, not members — so the
    // member-count guard only applies to Bonn talk groups.
    for (int i = 0; i < (int)tgProcs_.size(); ++i) {
        auto& p = tgProcs_[i];
        if (trafficMode_ != "padre" && (int)p.members.size() < 2) continue;
        p.state = TalkGroupProcess::State::CONV_IDLE;
        scheduleAt(simTime() + sampleConvIdle(), p.convIdleTimer);
    }
}

// ─── Conversation-process state machine ───────────────────────────────────────

// PADRE traffic: point-to-point partner selection for `sender`. With
// probability pCross_, prefers a partner in a different cluster (area);
// otherwise prefers the same cluster. Falls back to the other side if the
// preferred side has no candidates. Returns -1 if no partner exists at all
// (e.g. only one user total).
int RLBridge::pickPadrePartner(int sender)
{
    std::vector<int> cross, same;
    int senderArea = sampledUserAreas_[sender];
    for (int u = 0; u < numUsers_; ++u) {
        if (u == sender) continue;
        if (sampledUserAreas_[u] != senderArea) cross.push_back(u);
        else same.push_back(u);
    }

    bool wantCross = uniformTraffic() < pCross_;
    std::vector<int>& primary  = wantCross ? cross : same;
    std::vector<int>& fallback = wantCross ? same  : cross;
    std::vector<int>& candidates = !primary.empty() ? primary : fallback;

    if (candidates.empty()) return -1;
    return candidates[uniformIntTraffic(0, (int)candidates.size() - 1)];
}

void RLBridge::onConvIdleExpired(int tgId)
{
    auto& p = tgProcs_[tgId];

    if (trafficMode_ == "padre") {
        // Sender and partner are episode-fixed (chosen once in
        // buildPadreSenders). onCallEnd resets p.current/p.other to -1 when
        // a conversation fully ends, so restore them from the fixed fields
        // rather than re-picking — pickPadrePartner must not run again here.
        // ── $OM2G_DEMAND_RESHUFFLE_STEPS: NON-STATIONARY DEMAND ─────────────
        // Users are static and peer memory is sound, so with a fixed
        // sender->partner map ONE successful delivery is permanent knowledge:
        // the steady-state value of a state UPDATE is structurally near zero:
        // updates only have value when the state actually changes.
        //
        // With this set, each sender re-draws its partner every N simulated
        // steps, so a 250-step episode contains several demand epochs. A drone
        // that cannot hear its peers can only play the average demand; one
        // that can must re-target each epoch. This does not make the task
        // easier -- it makes the quantity being communicated time-varying,
        // which is what "state update" means. Default off: unset reproduces
        // every existing run byte-identical.
        static const int reshuffleSteps = [] {
            const char* e = std::getenv("OM2G_DEMAND_RESHUFFLE_STEPS");
            return e ? std::atoi(e) : 0;
        }();
        if (reshuffleSteps > 0 && trafficMode_ == "padre") {
            // stepCounter_ is the bridge's own RL step index -- no need to
            // convert simulated seconds, and it cannot drift from the env.
            const int epoch = stepCounter_ / reshuffleSteps;
            if (epoch != p.demandEpoch) {
                int np = pickPadrePartner(p.head);
                if (np >= 0) p.fixedPartner = np;
                p.demandEpoch = epoch;
            }
        }

        p.current   = p.head;
        p.other     = p.fixedPartner;
        p.isMesh    = false;
        p.initiator = -1;

        if (p.other < 0) {
            scheduleAt(simTime() + sampleConvIdle(), p.convIdleTimer);
            return;
        }

        startCall(tgId);
        return;
    }

    if ((int)p.members.size() < 2) {
        scheduleAt(simTime() + sampleConvIdle(), p.convIdleTimer);
        return;
    }

    std::vector<int> nonHead;
    for (int m : p.members) if (m != p.head) nonHead.push_back(m);
    if (nonHead.empty()) {
        scheduleAt(simTime() + sampleConvIdle(), p.convIdleTimer);
        return;
    }

    bool headInitiates = (uniformTraffic() < trafficParams_.headInitiatorProb);

    if (headInitiates) {
        p.current   = p.head;
        p.other     = nonHead[uniformIntTraffic(0, (int)nonHead.size() - 1)];
        p.isMesh    = false;
        p.initiator = -1;
    } else {
        int idx     = uniformIntTraffic(0, (int)nonHead.size() - 1);
        p.current   = nonHead[idx];
        p.initiator = p.current;

        if ((int)nonHead.size() >= 2 && uniformTraffic() < trafficParams_.meshProb) {
            p.isMesh = true;
            std::vector<int> others;
            for (int m : nonHead) if (m != p.current) others.push_back(m);
            p.other = others[uniformIntTraffic(0, (int)others.size() - 1)];
        } else {
            p.isMesh = false;
            p.other  = p.head;
        }
    }

    startCall(tgId);
}

void RLBridge::startCall(int tgId)
{
    auto& p = tgProcs_[tgId];
    p.state           = TalkGroupProcess::State::IN_CALL;
    activeSpeakers_[tgId] = p.current;

    double cht = sampleCallCHT();
    scheduleAt(simTime() + cht, p.callEndTimer);

    if (p.packetTimer->isScheduled()) cancelEvent(p.packetTimer);
    scheduleAt(simTime(), p.packetTimer);  // first packet immediately

    std::printf("CALL_START:%.6f,tg=%d,spk=%d,other=%d,mesh=%d\n",
                simTime().dbl(), tgId, p.current, p.other, p.isMesh ? 1 : 0);
    std::fflush(stdout);
}

void RLBridge::onCallEnd(int tgId)
{
    auto& p = tgProcs_[tgId];
    if (p.packetTimer->isScheduled()) cancelEvent(p.packetTimer);
    activeSpeakers_[tgId] = -1;

    std::printf("CALL_END:%.6f,tg=%d,spk=%d\n",
                simTime().dbl(), tgId, p.current);
    std::fflush(stdout);

    if (uniformTraffic() < trafficParams_.callContinuationProb) {
        // Short idle before next call in same conversation.
        p.state = TalkGroupProcess::State::BETWEEN_CALLS;
        scheduleAt(simTime() + sampleCallIdle(), p.callStartTimer);
    } else {
        // Conversation ends.
        p.current = -1; p.other = -1; p.initiator = -1;
        p.state   = TalkGroupProcess::State::CONV_IDLE;
        scheduleAt(simTime() + sampleConvIdle(), p.convIdleTimer);
    }
}

void RLBridge::onCallStart(int tgId)
{
    // Pick next speaker per Table 2 transition probabilities.
    auto& p = tgProcs_[tgId];

    if (trafficMode_ == "padre") {
        // PADRE traffic: sender is fixed, no speaker alternation.
        startCall(tgId);
        return;
    }

    int next;
    if (!p.isMesh) {
        if (p.current == p.head) {
            next = (uniformTraffic() < trafficParams_.transHeadToHead) ? p.head : p.other;
        } else {
            next = (uniformTraffic() < trafficParams_.transMemberToMember) ? p.current : p.head;
        }
    } else {
        // Mesh: initiator plays "H role".
        int H = p.initiator;
        if (p.current == H) {
            next = (uniformTraffic() < trafficParams_.transHeadToHead) ? p.current : p.other;
        } else {
            next = (uniformTraffic() < trafficParams_.transMemberToMember) ? p.current : H;
        }
    }

    if (next != p.current) std::swap(p.current, p.other);  // speaker switches
    startCall(tgId);
}

void RLBridge::onPacketTick(int tgId)
{
    auto& p = tgProcs_[tgId];
    if (p.state != TalkGroupProcess::State::IN_CALL || p.current < 0) return;

    if (p.current < (int)trafficApps_.size() && trafficApps_[p.current]) {
        bool padreOk = (trafficMode_ == "padre" && p.other >= 0
                        && p.other < (int)userIps_.size()
                        && !userIps_[p.other].isUnspecified());
        static int diagCount = 0;
        if (diagCount < 20) {
            std::printf("SEND_DIAG: mode=%s other=%d nUserIps=%d unspec=%d -> %s\n",
                        trafficMode_.c_str(), p.other, (int)userIps_.size(),
                        (p.other >= 0 && p.other < (int)userIps_.size())
                            ? (userIps_[p.other].isUnspecified() ? 1 : 0) : -1,
                        padreOk ? "UNICAST" : "MULTICAST");
            std::fflush(stdout);
            diagCount++;
        }
        if (padreOk)
            trafficApps_[p.current]->sendMelpe(userIps_[p.other]);   // unicast
        else
            trafficApps_[p.current]->sendMelpe(getTalkGroupMulticast(tgId));  // bonn
    }

    scheduleAt(simTime() + SimTime(trafficParams_.melpeIntervalMs, SIMTIME_MS), p.packetTimer);
}

// ─── RNG helpers ──────────────────────────────────────────────────────────────

double RLBridge::sampleConvIdle()
{
    double v = std::lognormal_distribution<double>(
                   trafficParams_.convIdle_meanlog,
                   trafficParams_.convIdle_sdlog)(trafficRng_)
               + trafficParams_.convIdle_shift;
    return std::max(0.01, v);
}

double RLBridge::sampleCallCHT()
{
    double v = std::lognormal_distribution<double>(
                   trafficParams_.callCHT_meanlog,
                   trafficParams_.callCHT_sdlog)(trafficRng_);
    return std::max(0.01, v);
}

double RLBridge::sampleCallIdle()
{
    double scale = 1.0 / trafficParams_.callIdle_rate;
    double v = std::gamma_distribution<double>(
                   trafficParams_.callIdle_shape, scale)(trafficRng_);
    return std::max(0.01, v);
}

double RLBridge::uniformTraffic()
{
    return std::uniform_real_distribution<double>(0.0, 1.0)(trafficRng_);
}

int RLBridge::uniformIntTraffic(int lo, int hi)
{
    return std::uniform_int_distribution<int>(lo, hi)(trafficRng_);
}

Ipv4Address RLBridge::getTalkGroupMulticast(int tgId) const
{
    if (tgId >= 0 && tgId <= 3)
        return Ipv4Address(("239.1.1." + std::to_string(tgId + 1)).c_str());
    return Ipv4Address("239.1.1.10");
}

// ─── Traffic-app registration ─────────────────────────────────────────────────

void RLBridge::registerTrafficApp(int userId, UserBonnTrafficApp* app,
                                  const TrafficParams& params)
{
    if (userId < 0 || userId >= numUsers_) return;
    trafficApps_[userId] = app;
    if (!trafficParams_.paramsLoaded)
        trafficParams_ = params;
}

// ─── Episode layout query helpers ────────────────────────────────────────────

Coord RLBridge::sampleInDisk(std::mt19937& rng, Coord center, double radius)
{
    double u     = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double theta = std::uniform_real_distribution<double>(0.0, 2.0 * M_PI)(rng);
    double r     = radius * std::sqrt(u);
    double x = std::max(0.0, std::min(arenaWidth_,  center.x + r * std::cos(theta)));
    double y = std::max(0.0, std::min(arenaHeight_, center.y + r * std::sin(theta)));
    return Coord(x, y, 0.0);
}

Coord RLBridge::getSampledUserPosition(int userId) const
{
    if (userId < 0 || userId >= numUsers_) return Coord::ZERO;
    return sampledUserPositions_[userId];
}

int RLBridge::getUserArea(int userId) const
{
    if (userId < 0 || userId >= numUsers_) return 0;
    return sampledUserAreas_[userId];
}

// ─── Drone registration / association ────────────────────────────────────────

void RLBridge::registerDrone(int droneId, cModule* mobilityModule)
{
    if (droneId < 0 || droneId >= numDrones_)
        throw cRuntimeError("RLBridge: droneId %d out of range", droneId);
    drones_[droneId] = mobilityModule;
}

void RLBridge::getAction(int droneId, double& outDx, double& outDy) const
{
    const auto& a = actionBuffer_.at(droneId);
    outDx = a.first; outDy = a.second;
}

// ── OM2G: inter-agent state-message accounting ───────────────────────────────
// Called by DroneBeaconApp on the REAL UDP multicast path.  noteBeaconReceived
// fires only when a datagram is fully reassembled and delivered to the socket,
// so a message lost to a collision, to a range failure, or to a single missing
// IPv4 fragment leaves beaconRxTime_[i][j] untouched and observer i keeps
// seeing a stale peer.  That is the whole point of the track: the compression
// ratio decides how many fragments a message costs, and the fragment count
// decides how often it survives.
void RLBridge::noteBeaconSent(int txDroneId, int payloadBytes)
{
    if (txDroneId < 0 || txDroneId >= numDrones_) return;
    ctrlBytesTx_[txDroneId] += payloadBytes;
    beaconTxCount_[txDroneId] += 1;
    beaconPayloadBytes_ = payloadBytes;
}

void RLBridge::noteBeaconReceived(int rxDroneId, int txDroneId, int payloadBytes)
{
    if (rxDroneId < 0 || rxDroneId >= numDrones_) return;
    ctrlBytesRx_[rxDroneId] += payloadBytes;
    beaconRxCount_[rxDroneId] += 1;
    if (txDroneId < 0 || txDroneId >= numDrones_) return;
    beaconRxTime_[rxDroneId][txDroneId]  = simTime();
    beaconRxValid_[rxDroneId][txDroneId] = 1.0;
}

void RLBridge::noteForwardedPeerState(int rxDroneId, int aboutDroneId,
                                      double originTimeSec)
{
    if (rxDroneId < 0 || rxDroneId >= numDrones_) return;
    if (aboutDroneId < 0 || aboutDroneId >= numDrones_) return;
    if (aboutDroneId == rxDroneId) return;          // never learn about self
    // Only accept a forward that is NEWER than what we already hold, so a
    // relayed copy can never make our picture staler, and so flooding
    // converges instead of oscillating.
    SimTime origin = SimTime(originTimeSec);
    if (beaconRxValid_[rxDroneId][aboutDroneId] > 0.5
            && beaconRxTime_[rxDroneId][aboutDroneId] >= origin)
        return;
    beaconRxTime_[rxDroneId][aboutDroneId]  = origin;
    beaconRxValid_[rxDroneId][aboutDroneId] = 1.0;
}

int RLBridge::getKnownPeerStates(int droneId, std::vector<int>& ids,
                                 std::vector<double>& originTimes) const
{
    ids.clear(); originTimes.clear();
    if (droneId < 0 || droneId >= numDrones_) return 0;
    for (int j = 0; j < numDrones_; ++j) {
        if (j == droneId) continue;
        if (beaconRxValid_[droneId][j] <= 0.5) continue;
        ids.push_back(j);
        originTimes.push_back(beaconRxTime_[droneId][j].dbl());
    }
    return (int)ids.size();
}

int RLBridge::getUserCount(int droneId) const
{
    if (droneId < 0 || droneId >= numDrones_) return 0;
    return droneUserCounts_[droneId];
}

int RLBridge::getSensedUserCount(int droneId) const
{
    if (droneId < 0 || droneId >= numDrones_) return 0;
    int count = 0;
    for (double v : userRssiValid_[droneId])
        if (v > 0.5) ++count;
    return count;
}

bool RLBridge::getLearnedGateWantsSend(int droneId) const
{
    if (droneId < 0 || droneId >= (int)sendGateBuffer_.size()) return true;
    return sendGateBuffer_[droneId] > 0.0;
}

void RLBridge::getDronePosition(int droneId, double& outX, double& outY) const
{
    outX = 0.0; outY = 0.0;
    if (droneId < 0 || droneId >= numDrones_ || !drones_[droneId]) return;
    Coord p = check_and_cast<MobilityBase*>(drones_[droneId])->getCurrentPosition();
    outX = p.x; outY = p.y;
}

void RLBridge::resolveDroneIps()
{
    cModule* net = getParentModule();
    droneIps_.assign(numDrones_, Ipv4Address::UNSPECIFIED_ADDRESS);
    for (int d = 0; d < numDrones_; ++d) {
        cModule* h = net->getSubmodule("host", d);
        if (!h) continue;
        auto* ift = L3AddressResolver().findInterfaceTableOf(h);
        if (!ift) continue;
        L3Address a = L3AddressResolver().getAddressFrom(ift);
        if (!a.isUnspecified()) droneIps_[d] = a.toIpv4();
    }
}

// ── A2 oracle routing ────────────────────────────────────────────────────────
// Recompute every drone's /32 routes to every associated user, from scratch.
//
// Drone graph: edge (i,j) iff dist(i,j) <= DRONE_LINK_RANGE_M. BFS from each
// drone gives the shortest path to every other drone; the FIRST hop of that
// path is the gateway. So every installed route's gateway is a drone that is
// genuinely one L2 hop away -- INET's routeUnicastPacket ARPs the gateway
// directly and does NOT recurse, so a non-adjacent gateway would be a silent
// black hole. That is the invariant this function exists to guarantee.
//
// The drone a user is attached to installs the /32 with NO gateway: that user
// really is L2-adjacent to it, so on-link delivery is correct there.
void RLBridge::rebuildOracleRoutes()
{
    if ((int)droneIps_.size() != numDrones_) resolveDroneIps();
    cModule* net = getParentModule();

    // adjacency
    std::vector<std::vector<bool>> adj(numDrones_, std::vector<bool>(numDrones_, false));
    for (int i = 0; i < numDrones_; ++i)
        for (int j = 0; j < numDrones_; ++j) {
            if (i == j) continue;
            double ix, iy, jx, jy;
            getDronePosition(i, ix, iy);
            getDronePosition(j, jx, jy);
            double dx = ix - jx, dy = iy - jy;
            adj[i][j] = (std::sqrt(dx*dx + dy*dy) <= DRONE_LINK_RANGE_M);
        }

    // firstHop[s][t] = first drone on the shortest path s -> t, or -1 if none
    std::vector<std::vector<int>> firstHop(numDrones_, std::vector<int>(numDrones_, -1));
    for (int s = 0; s < numDrones_; ++s) {
        std::vector<int> prev(numDrones_, -1);
        std::vector<bool> seen(numDrones_, false);
        std::deque<int> q;
        seen[s] = true; q.push_back(s);
        while (!q.empty()) {
            int u = q.front(); q.pop_front();
            for (int v = 0; v < numDrones_; ++v)
                if (adj[u][v] && !seen[v]) { seen[v] = true; prev[v] = u; q.push_back(v); }
        }
        for (int t = 0; t < numDrones_; ++t) {
            if (t == s || !seen[t]) continue;
            int cur = t;                       // walk back to the hop after s
            while (prev[cur] != s) cur = prev[cur];
            firstHop[s][t] = cur;
        }
    }

    // GRAPH_DIAG: connected components of the drone adjacency graph.
    // With r_d = 523 m the drone graph is not complete by construction. When a
    // policy scatters the drones the graph fragments, BFS finds no path, and
    // packets addressed behind unreachable drones are dropped after ARP retries.
    // comps > 1 identifies such drops as a GEOMETRY effect (spawn/arena).
    {
        std::vector<int> comp(numDrones_, -1);
        int nComp = 0, largest = 0;
        for (int s = 0; s < numDrones_; ++s) {
            if (comp[s] != -1) continue;
            int sz = 0;
            std::deque<int> q; q.push_back(s); comp[s] = nComp;
            while (!q.empty()) {
                int u = q.front(); q.pop_front(); ++sz;
                for (int v = 0; v < numDrones_; ++v)
                    if (adj[u][v] && comp[v] == -1) { comp[v] = nComp; q.push_back(v); }
            }
            if (sz > largest) largest = sz;
            ++nComp;
        }
        int unreachPairs = 0;
        for (int i = 0; i < numDrones_; ++i)
            for (int j = 0; j < numDrones_; ++j)
                if (i != j && firstHop[i][j] < 0) ++unreachPairs;
        graphComps_   = nComp;
        graphLargest_ = largest;
        graphUnreach_ = unreachPairs;
    }

    int reachable = 0, total = 0;

    for (int d = 0; d < numDrones_; ++d) {
        cModule* h = net->getSubmodule("host", d);
        if (!h) continue;
        auto* irt = check_and_cast<IIpv4RoutingTable*>(h->getModuleByPath(".ipv4.routingTable"));
        auto* ift = check_and_cast<IInterfaceTable*>(h->getSubmodule("interfaceTable"));
        NetworkInterface* wlan = ift->findInterfaceByName("wlan0");
        if (!wlan) continue;

        // wipe every /32 we own (MANUAL + IFACENETMASK), leave AODV/loopback alone
        for (int i = 0; i < irt->getNumRoutes(); ) {
            Ipv4Route* r = irt->getRoute(i);
            bool ours = (r->getNetmask() == Ipv4Address::ALLONES_ADDRESS)
                     && (r->getSourceType() == IRoute::MANUAL
                      || r->getSourceType() == IRoute::IFACENETMASK);
            if (ours) irt->deleteRoute(r); else ++i;
        }

        for (int u = 0; u < numUsers_; ++u) {
            int home = userToDrone_[u];
            if (home < 0 || home >= numDrones_) continue;      // not associated yet
            if (userIps_[u].isUnspecified()) continue;
            ++total;

            auto* r = new Ipv4Route();
            r->setDestination(userIps_[u]);
            r->setNetmask(Ipv4Address::ALLONES_ADDRESS);
            r->setInterface(wlan);

            if (d == home) {
                // user is on my own cell: genuinely on-link, no gateway
                r->setSourceType(IRoute::IFACENETMASK);
                r->setMetric(0);
                irt->addRoute(r);
                ++reachable;
            }
            else {
                int nh = firstHop[d][home];
                if (nh < 0 || droneIps_[nh].isUnspecified()) {
                    delete r;                                   // no path: install nothing
                    continue;
                }
                r->setGateway(droneIps_[nh]);                   // an ADJACENT drone
                r->setSourceType(IRoute::MANUAL);
                r->setMetric(1);
                irt->addRoute(r);
                ++reachable;
            }
        }
    }

    routeAuditReachable_ = reachable;
    routeAuditTotal_     = total;
}

void RLBridge::notifyAllDronesOfAssociation(int userId, Ipv4Address userIp,
                                             int newDroneId, int oldDroneId)
{
    if (oldDroneId >= 0 && oldDroneId < numDrones_)
        droneUserCounts_[oldDroneId] = std::max(0, droneUserCounts_[oldDroneId] - 1);
    if (newDroneId >= 0 && newDroneId < numDrones_)
        droneUserCounts_[newDroneId]++;
    if (userId >= 0 && userId < numUsers_) {
        userToDrone_[userId] = newDroneId;
        userIps_[userId]     = userIp;
    }
    rebuildOracleRoutes();
}

// ─── Sync cycle ───────────────────────────────────────────────────────────────

void RLBridge::handleMessage(cMessage* msg)
{
    if (msg == syncEvent_) {
        if (stepCounter_ == 0)
            startConversationProcesses();

        updateDronePairLines();
        rebuildOracleRoutes();          // drones moved: adjacency may have changed
        std::printf("GRAPH_DIAG: step=%d comps=%d largest=%d/%d unreach_pairs=%d/%d routes=%d/%d\n",
                    stepCounter_, graphComps_, graphLargest_, numDrones_,
                    graphUnreach_, numDrones_ * (numDrones_ - 1),
                    routeAuditReachable_, routeAuditTotal_);
        std::fflush(stdout);

        // ── POS_DIAG: collapse detector (see patch_posdiag.py) ─────────────
        if (stepCounter_ % 25 == 0) {
            std::vector<double> px(numDrones_), py(numDrones_);
            for (int d = 0; d < numDrones_; ++d)
                getDronePosition(d, px[d], py[d]);

            double cx = 0.0, cy = 0.0;
            for (int d = 0; d < numDrones_; ++d) { cx += px[d]; cy += py[d]; }
            cx /= numDrones_; cy /= numDrones_;

            double rms = 0.0;
            int edge = 0;
            for (int d = 0; d < numDrones_; ++d) {
                double ex = px[d] - cx, ey = py[d] - cy;
                rms += ex * ex + ey * ey;
                double wall = std::min(std::min(px[d], py[d]),
                                       std::min(arenaWidth_ - px[d],
                                                arenaHeight_ - py[d]));
                if (wall < 150.0) edge++;
            }
            rms = std::sqrt(rms / numDrones_);

            // Common-mode coherence over the window. 1.0 == rigid-body
            // translation (the runaway); ~0.35 == uncorrelated motion at n=8.
            double sdx = 0.0, sdy = 0.0, smag = 0.0, coh = 0.0;
            if ((int)posPrevX_.size() == numDrones_) {
                for (int d = 0; d < numDrones_; ++d) {
                    double dx = px[d] - posPrevX_[d];
                    double dy = py[d] - posPrevY_[d];
                    sdx += dx; sdy += dy;
                    smag += std::sqrt(dx * dx + dy * dy);
                }
                if (smag > 1e-9)
                    coh = std::sqrt(sdx * sdx + sdy * sdy) / smag;
            }
            posPrevX_ = px; posPrevY_ = py;

            std::printf("POS_DIAG: step=%d cx=%.0f cy=%.0f rms=%.0f "
                        "edge=%d/%d coh=%.2f drift=%.0f\n",
                        stepCounter_, cx, cy, rms, edge, numDrones_, coh,
                        std::sqrt(sdx * sdx + sdy * sdy));
            std::fflush(stdout);
        }
        if (stepCounter_ % 100 == 0) {
            std::printf("ROUTE_AUDIT: step=%d installed=%d/%d range=%.0fm\n",
                        stepCounter_, routeAuditReachable_, routeAuditTotal_,
                        DRONE_LINK_RANGE_M);
            std::fflush(stdout);
        }
        printAndResetEventCounts();

        bool done = (stepCounter_ >= maxSteps_);
        std::string obsJson = buildObservationJson(done);

        zmq::message_t req(obsJson.data(), obsJson.size());
        sock_->send(req, zmq::send_flags::none);

        zmq::message_t rep;
        auto result = sock_->recv(rep, zmq::recv_flags::none);
        if (!result) throw cRuntimeError("RLBridge: ZMQ recv failed");
        std::string actionJson(static_cast<const char*>(rep.data()), rep.size());

        if (!done) parseActionJson(actionJson);

        // ── $OM2G_USER_ROAM_M: MOBILE USERS ────────────────────────────────
        // With static users and peer memory, ONE successful delivery is
        // permanent knowledge, so the steady-state value of a state UPDATE is
        // structurally ~0. State UPDATES crowd the network, and an update only
        // has value when the state actually CHANGES.
        //
        // Each user takes a small random step per RL step, tethered to within
        // OM2G_USER_ROAM_M of where it spawned. Tethering matters: cluster
        // identity has to survive (the demand tags and the whole archipelago
        // geometry depend on it) while the exact positions go stale. People
        // milling around a site, not evacuating it.
        //
        // Default 0 = perfectly static = every existing run reproduced exactly.
        static const double roamM = [] {
            const char* e = std::getenv("OM2G_USER_ROAM_M");
            return e ? std::atof(e) : 0.0;
        }();
        static const double stepM = [] {
            const char* e = std::getenv("OM2G_USER_SPEED_MPS");
            return e ? std::atof(e) : 8.0;   // metres per RL step
        }();
        if (!done && roamM > 0.0 && numUsers_ > 0) {
            if ((int)userHome_.size() != numUsers_)
                userHome_ = sampledUserPositions_;   // first step: remember spawn
            for (int u = 0; u < numUsers_; ++u) {
                const double ang = uniformTraffic() * 2.0 * M_PI;
                Coord np = sampledUserPositions_[u]
                         + Coord(stepM * std::cos(ang), stepM * std::sin(ang), 0.0);
                // tether: never further than roamM from home
                const double dx = np.x - userHome_[u].x, dy = np.y - userHome_[u].y;
                const double d  = std::sqrt(dx * dx + dy * dy);
                if (d > roamM) {
                    np.x = userHome_[u].x + dx * roamM / d;
                    np.y = userHome_[u].y + dy * roamM / d;
                }
                np.x = std::min(arenaWidth_,  std::max(0.0, np.x));
                np.y = std::min(arenaHeight_, std::max(0.0, np.y));
                sampledUserPositions_[u] = np;
            }
        }

        stepCounter_++;
        if (!done) scheduleAt(simTime() + syncInterval_, syncEvent_);
        return;
    }

    // Dispatch talk-group timer messages.
    for (int i = 0; i < (int)tgProcs_.size(); ++i) {
        auto& p = tgProcs_[i];
        if (msg == p.convIdleTimer)  { onConvIdleExpired(i); return; }
        if (msg == p.callEndTimer)   { onCallEnd(i); return; }
        if (msg == p.callStartTimer) { onCallStart(i); return; }
        if (msg == p.packetTimer)    { onPacketTick(i); return; }
    }

    throw cRuntimeError("RLBridge: unexpected message '%s'", msg->getName());
}

// ─── Diagnostic MAC/PHY event-count measurement ──────────────────────────────

void RLBridge::printAndResetEventCounts()
{
    for (int i = 0; i < numDrones_; ++i) {
        auto& c = droneEventCounts_[i];
        std::printf("[EVENT_COUNT] step=%d drone=%d rx=%ld drop_retry=%ld "
                    "drop_queue=%ld drop_corrupt=%ld drop_notforus=%ld link_broken=%ld "
                    "net_drop_no_route=%ld net_drop_addr_res_failed=%ld net_drop_hop_limit=%ld "
                    "net_drop_fwd_disabled=%ld net_drop_no_iface=%ld net_drop_lifetime=%ld "
                    "net_drop_checksum=%ld net_drop_undefined=%ld event_overflow=%ld\n",
                    stepCounter_, i, c.rx, c.dropRetry, c.dropQueue,
                    c.dropCorrupt, c.dropNotForUs, c.linkBroken,
                    c.netDropNoRoute, c.netDropAddrResFailed, c.netDropHopLimit,
                    c.netDropFwdDisabled, c.netDropNoIface, c.netDropLifetime,
                    c.netDropChecksum, c.netDropUndefined, c.eventOverflow);
        c = EventCounts{};
    }
    std::fflush(stdout);
}

// ─── Raw per-event MAC/PHY observation set ───────────────────────────────────

void RLBridge::recordEvent(int droneIdx, const EventRecord& rec)
{
    if (droneIdx < 0 || droneIdx >= (int)droneEvents_.size()) return;

    // CUT: drop NOT_ADDRESSED_TO_US before it can consume a cap slot. This
    // MUST happen before the cap check, not after — at ~480 raw events/step
    // against a cap of 32, filtering downstream means the surviving 32 are
    // just the first 32 to arrive (~87% of them not-for-us), so real
    // receptions are lost by arrival order. The frozen event encoder
    // (h64_o8) was trained on the cut set; feeding it anything else is a
    // silent distribution break. Counted as an overflow-class discard so the
    // rate stays visible in [EVENT_COUNT].
    if (cutNotForUs_ && rec.dropNotForUs != 0.0) return;

    auto& v = droneEvents_[droneIdx];
    if ((int)v.size() >= MAX_EVENTS_CAP) {
        droneEventCounts_[droneIdx].eventOverflow++;
        return;
    }
    v.push_back(rec);
}

// ─── Observation JSON ─────────────────────────────────────────────────────────

std::string RLBridge::buildObservationJson(bool done)
{
    json j;
    j["type"] = "obs";
    j["step"] = stepCounter_;
    j["done"] = done;
    // Scales for normalisation: the drone token's dx/dy are metres (up to
    // +-1700) and time_since_rx is seconds (up to 1000); fed raw into a
    // randomly-initialised actor they overflow (DeepSets sums over peers, ST
    // softmaxes) and produce NaN. Python needs these scales to normalise -- they
    // are emitted here rather than hardcoded on the Python side so the
    // simulator stays the single source of these parameters.
    j["arena_width"]  = arenaWidth_;
    j["arena_height"] = arenaHeight_;
    j["max_steps"]    = maxSteps_;

    // Drone positions.
    json positions = json::array();
    for (int i = 0; i < numDrones_; ++i) {
        if (!drones_[i]) { positions.push_back({0.0, 0.0}); continue; }
        Coord p = check_and_cast<MobilityBase*>(drones_[i])->getCurrentPosition();
        positions.push_back({p.x, p.y});
    }
    j["positions"] = positions;

    // Drone throughput.
    double dt = syncInterval_.dbl();
    json throughputs = json::array(), drone_up = json::array(), drone_fwd = json::array();
    for (int i = 0; i < numDrones_; ++i) {
        throughputs.push_back(droneBytesRx_[i] / dt);
        drone_up.push_back(droneBytesUp_[i] / dt);
        drone_fwd.push_back((droneBytesRx_[i] - droneBytesUp_[i]) / dt);
        droneBytesRx_[i] = 0L; droneBytesUp_[i] = 0L;
    }
    j["throughput"] = throughputs;
    j["drone_up"]   = drone_up;
    j["drone_fwd"]  = drone_fwd;

    // ── OM2G: control-plane vs data-plane, and state-message delivery ────────
    // Inert on the OMNET track (omnet_env.py never reads these keys); consumed
    // by om2g_env.py.  beacon_rx_age[i][j] is seconds since observer i last
    // received a state message from peer j -- the staleness that gates whether
    // i may use j's state at all.
    {
        SimTime nowB = simTime();
        json ctrlTx = json::array(), ctrlRx = json::array();
        json bcnTx = json::array(), bcnRx = json::array();
        for (int i = 0; i < numDrones_; ++i) {
            ctrlTx.push_back(ctrlBytesTx_[i] / dt);
            ctrlRx.push_back(ctrlBytesRx_[i] / dt);
            bcnTx.push_back(beaconTxCount_[i]);
            bcnRx.push_back(beaconRxCount_[i]);
            ctrlBytesTx_[i] = 0L; ctrlBytesRx_[i] = 0L;
            beaconTxCount_[i] = 0L; beaconRxCount_[i] = 0L;
        }
        j["ctrl_bytes_tx"]  = ctrlTx;   // control-plane bytes/s offered
        j["ctrl_bytes_rx"]  = ctrlRx;   // control-plane bytes/s delivered
        j["beacon_tx_count"] = bcnTx;
        j["beacon_rx_count"] = bcnRx;
        j["beacon_payload_bytes"] = beaconPayloadBytes_;

        json bAge = json::array(), bValid = json::array();
        for (int i = 0; i < numDrones_; ++i) {
            json rowA = json::array(), rowV = json::array();
            for (int jd = 0; jd < numDrones_; ++jd) {
                rowA.push_back(beaconRxValid_[i][jd] > 0.0
                               ? (nowB - beaconRxTime_[i][jd]).dbl() : -1.0);
                rowV.push_back(beaconRxValid_[i][jd]);
            }
            bAge.push_back(std::move(rowA));
            bValid.push_back(std::move(rowV));
        }
        j["beacon_rx_age"]   = bAge;    // seconds; -1.0 = never received
        j["beacon_rx_valid"] = bValid;
    }

    // User positions.
    json userPositions = json::array();
    for (int i = 0; i < numUsers_; ++i)
        userPositions.push_back({sampledUserPositions_[i].x, sampledUserPositions_[i].y});
    j["user_positions"] = userPositions;

    // NEW: user_area int[N_users] with values in {0,1,2,3}.
    j["user_area"] = sampledUserAreas_;

    // sender_pairs: the FIXED sender->partner assignments drawn once at episode
    // start (buildPadreSenders). These are the conversations the radio actually
    // carries -- ~numUsers of them, NOT all N*(N-1) ordered pairs.
    //
    // Without this the reward would have to build commodities for EVERY ordered
    // user pair (17 users -> 272 commodities): an LP ~16x larger than the task
    // requires, weighting cross-cluster traffic at 66% by combinatorics rather
    // than at the traffic model's own 50%.
    //
    // fixedPartner, NOT other: `other` is reset to -1 by onCallEnd when a
    // conversation ends, so keying the reward on it would make the reward BURSTY
    // -- a function of who happens to be mid-call, not of drone positions. That is
    // what SS6.3 rules out. fixedPartner is stable for
    // the whole episode.
    {
        json pairs = json::array();
        for (const auto& p : tgProcs_) {
            if (p.head < 0 || p.fixedPartner < 0) continue;
            pairs.push_back(json::array({p.head, p.fixedPartner}));
        }
        j["sender_pairs"] = pairs;
    }

    // NEW: active_broadcasts — list of {talk_group_id, speaker_user_id} for active calls.
    json abArray = json::array();
    for (int t = 0; t < (int)activeSpeakers_.size(); ++t) {
        if (activeSpeakers_[t] >= 0) {
            json ab;
            ab["talk_group_id"]   = t;
            ab["speaker_user_id"] = activeSpeakers_[t];
            abArray.push_back(std::move(ab));
        }
    }
    j["active_broadcasts"] = abArray;

    // NEW: talk_group_heads — int[5].
    j["talk_group_heads"] = talkGroupHeads_;

    // Zone centers (kept for backward compat).
    json zoneCenters;
    zoneCenters["command"] = {commandZoneCenter_.x, commandZoneCenter_.y};
    zoneCenters["patients_waiting"] = {patientsWaitingCenter_.x, patientsWaitingCenter_.y};
    json incCenters = json::array();
    for (auto& c : incidentZoneCenters_)
        incCenters.push_back({c.x, c.y});
    zoneCenters["incidents"] = incCenters;
    json ccsCenters = json::array();
    for (auto& c : ccsZoneCenters_)
        ccsCenters.push_back({c.x, c.y});
    zoneCenters["ccs_zones"] = ccsCenters;
    j["zone_centers"]     = zoneCenters;
    j["ccs_zone_centers"] = ccsCenters;
    j["num_incidents"]    = numSampledIncidents_;

    // Per-user goodput.
    json goodput = json::array(), user_up = json::array();
    for (int i = 0; i < numUsers_; ++i) {
        goodput.push_back(userBytesRx_[i] / dt);
        user_up.push_back(userBytesUp_[i] / dt);
        userBytesRx_[i] = 0L; userBytesUp_[i] = 0L;
    }
    j["user_goodput"] = goodput;
    j["user_up"]      = user_up;
    j["user_to_drone"] = userToDrone_;

    j["drone_user_counts"] = droneUserCounts_;

    // ── noise_floor: received interference power from jammer, one entry per drone ──
    // Computed analytically with deterministic log-distance path loss (alpha=2.5,
    // 2.4 GHz, d0=1 m) — no shadowing draw, so the value is stable across steps.
    // Before step `activate`: returns ambient floor (-100 dBm).
    // After  step `activate`: returns jammer-induced interference in dBm.
    {
        static const double kC       = 3e8;
        static const double kF       = 2.4e9;
        static const double kLambda  = kC / kF;
        static const double kLossD0  = (kLambda * kLambda) / (16.0 * M_PI * M_PI);
        static const double kAlpha   = 2.5;
        static const double kTxW     = 0.3;   // 300 mW jammer TX power
        static const double kAmbient = -100.0; // dBm ambient before activation

        json noiseFloor = json::array();

        bool jammerActive = (jammerModule_ != nullptr && stepCounter_ >= jammerActivateStep_);

        if (jammerActive) {
            auto* mob = check_and_cast<MobilityBase*>(
                            jammerModule_->getSubmodule("mobility"));
            Coord jPos = mob->getCurrentPosition();

            for (int i = 0; i < numDrones_; ++i) {
                if (!drones_[i]) { noiseFloor.push_back(kAmbient); continue; }
                Coord dPos = check_and_cast<MobilityBase*>(drones_[i])->getCurrentPosition();
                double d = jPos.distance(dPos);
                if (d < 1.0) d = 1.0;
                double loss    = kLossD0 / std::pow(d, kAlpha);
                double rxW     = kTxW * loss;
                double rxdBm   = 10.0 * std::log10(rxW * 1000.0); // W→mW→dBm
                noiseFloor.push_back(rxdBm);
            }

            // Verification log: jammer position + per-drone noise_floor.
            std::printf("[JAMMER_VERIFY] step=%d jammer=(%.1f,%.1f) noise_floor_dBm=[",
                        stepCounter_, jPos.x, jPos.y);
            for (int i = 0; i < numDrones_; ++i)
                std::printf("%.1f%s", noiseFloor[i].get<double>(),
                            (i + 1 < numDrones_) ? "," : "");
            std::printf("]\n");
            std::fflush(stdout);
        } else {
            for (int i = 0; i < numDrones_; ++i)
                noiseFloor.push_back(kAmbient);

            // Log jammer position even before activation (for convergence check).
            if (jammerModule_) {
                auto* mob = check_and_cast<MobilityBase*>(
                                jammerModule_->getSubmodule("mobility"));
                Coord jPos = mob->getCurrentPosition();
                std::printf("[JAMMER_VERIFY] step=%d jammer=(%.1f,%.1f) noise_floor=ambient\n",
                            stepCounter_, jPos.x, jPos.y);
                std::fflush(stdout);
            }
        }

        j["noise_floor"] = noiseFloor;
    }

    // ── interference: ANALYTIC received power at each drone from all OTHER
    // transmitters (every user + every other drone), one entry per drone.
    // Same log-distance path-loss model as noise_floor (alpha=2.5, 2.4 GHz,
    // d0=1 m), TX powers: drones 200 mW, users 10 mW. Assumes ALL other
    // transmitters are concurrently active — this ignores CSMA scheduling
    // (802.11 defers most of the time), so it is an upper-bound/analytic
    // estimate, not a measured value, exactly like noise_floor. The jammer is
    // EXCLUDED here (it already lives in noise_floor and stays separate).
    {
        static const double kC       = 3e8;
        static const double kF       = 2.4e9;
        static const double kLambda  = kC / kF;
        static const double kLossD0  = (kLambda * kLambda) / (16.0 * M_PI * M_PI);
        static const double kAlpha   = 2.5;
        static const double kTxMwDrone = 200.0;
        static const double kTxMwUser  = 10.0;

        json interference = json::array();

        std::vector<Coord> dronePos(numDrones_);
        std::vector<bool>  droneLive(numDrones_, false);
        for (int i = 0; i < numDrones_; ++i) {
            if (!drones_[i]) continue;
            dronePos[i]  = check_and_cast<MobilityBase*>(drones_[i])->getCurrentPosition();
            droneLive[i] = true;
        }

        for (int i = 0; i < numDrones_; ++i) {
            if (!droneLive[i]) { interference.push_back(-100.0); continue; }

            double sumMw = 0.0;

            for (int u = 0; u < numUsers_; ++u) {
                double d = dronePos[i].distance(sampledUserPositions_[u]);
                if (d < 1.0) d = 1.0;
                double loss = kLossD0 / std::pow(d, kAlpha);
                sumMw += kTxMwUser * loss;
            }

            for (int j = 0; j < numDrones_; ++j) {
                if (j == i || !droneLive[j]) continue;
                double d = dronePos[i].distance(dronePos[j]);
                if (d < 1.0) d = 1.0;
                double loss = kLossD0 / std::pow(d, kAlpha);
                sumMw += kTxMwDrone * loss;
            }

            interference.push_back(sumMw > 0.0 ? 10.0 * std::log10(sumMw) : -100.0);
        }

        j["interference"] = interference;
    }

    // Talk group memberships (updated to reflect new 5-group structure).
    json tgArray = json::array();
    for (auto& tg : talkGroups_) {
        json t;
        t["type"]     = tg.type;
        t["incident"] = tg.incident;
        t["head"]     = tg.head;
        t["members"]  = tg.members;
        tgArray.push_back(std::move(t));
    }
    j["talk_group_memberships"] = tgArray;

    // ── events: per-drone raw per-event MAC/PHY observation set (see
    // RLBridge.h EventRecord for the 12-column layout). Emitted as only the
    // REAL events accumulated since the previous step — no padding here;
    // Python pads to MAX_EVENTS_CAP. Cleared immediately after being read
    // into JSON so the next RL step starts from an empty accumulator.
    {
        json eventsArray = json::array();
        for (int i = 0; i < numDrones_; ++i) {
            json droneEventsJson = json::array();
            for (auto& e : droneEvents_[i]) {
                droneEventsJson.push_back({
                    e.snir, e.snirValid, e.per, e.ber, e.size, e.isDrop,
                    e.dropRetry, e.dropQueue, e.dropCorrupt, e.dropNotForUs,
                    e.fromDrone, e.present
                });
            }
            eventsArray.push_back(std::move(droneEventsJson));
            droneEvents_[i].clear();
        }
        j["events"] = eventsArray;
    }

    // ── peer_rssi / user_rssi: MEASURED received power (dBm), read straight
    // off arriving frames (SignalPowerInd) — NOT computed from positions. See
    // RLBridge.h peerDroneRssiDbm_/userRssiDbm_ comments. A drone that has
    // never received a frame from a given peer/user reports valid=0.0 and
    // dBm=0.0 — that silence is the actual sensor state, not a value to be
    // imputed downstream.
    {
        SimTime now = simTime();
        json peerRssi = json::array(), peerRssiValid = json::array(), peerRssiAge = json::array();
        for (int i = 0; i < numDrones_; ++i) {
            json row = json::array(), rowValid = json::array(), rowAge = json::array();
            for (int jd = 0; jd < numDrones_; ++jd) {
                row.push_back(peerDroneRssiDbm_[i][jd]);
                rowValid.push_back(peerDroneRssiValid_[i][jd]);
                rowAge.push_back((now - peerDroneRssiTime_[i][jd]).dbl());
            }
            peerRssi.push_back(std::move(row));
            peerRssiValid.push_back(std::move(rowValid));
            peerRssiAge.push_back(std::move(rowAge));
        }
        j["peer_rssi"]       = peerRssi;
        j["peer_rssi_valid"] = peerRssiValid;
        j["peer_rssi_age"]   = peerRssiAge;

        json userRssi = json::array(), userRssiValidJ = json::array(),
             userRssiAge = json::array();
        for (int i = 0; i < numDrones_; ++i) {
            json row = json::array(), rowValid = json::array(), rowAge = json::array();
            for (int u = 0; u < numUsers_; ++u) {
                row.push_back(userRssiDbm_[i][u]);
                rowValid.push_back(userRssiValid_[i][u]);
                // Seconds since this drone last received a frame FROM this user.
                // valid=1 with a large age means "I met them once and they may
                // have walked away since" -- which is a different epistemic
                // state from "I can hear them right now", and until this line
                // existed the two were indistinguishable downstream.
                rowAge.push_back((now - userRssiTime_[i][u]).dbl());
            }
            userRssi.push_back(std::move(row));
            userRssiValidJ.push_back(std::move(rowValid));
            userRssiAge.push_back(std::move(rowAge));
        }
        j["user_rssi"]       = userRssi;
        j["user_rssi_valid"] = userRssiValidJ;
        j["user_rssi_age"]   = userRssiAge;
    }

    // ── channel_busy / time_since_rx: raw own-state radio sensors, one entry
    // per drone (see RLBridge.h droneBusyTime_/droneLastRxTime_ comments).
    // channel_busy finalizes the accumulator up through "now" (there may be
    // no transition exactly at the step boundary) and resets it, exactly
    // parallel to droneEvents_ above. time_since_rx is not reset — it is a
    // running "seconds since" value by construction.
    {
        SimTime now = simTime();
        json channelBusy = json::array(), timeSinceRx = json::array();
        for (int i = 0; i < numDrones_; ++i) {
            bool wasBusy = (droneReceptionState_[i] == physicallayer::IRadio::RECEPTION_STATE_BUSY ||
                            droneReceptionState_[i] == physicallayer::IRadio::RECEPTION_STATE_RECEIVING);
            if (wasBusy)
                droneBusyTime_[i] += (now - droneLastStateChangeTime_[i]).dbl();
            droneLastStateChangeTime_[i] = now;

            channelBusy.push_back(dt > 0.0 ? droneBusyTime_[i] / dt : 0.0);
            droneBusyTime_[i] = 0.0;

            timeSinceRx.push_back((now - droneLastRxTime_[i]).dbl());
        }
        j["channel_busy"]  = channelBusy;
        j["time_since_rx"] = timeSinceRx;
    }

    return j.dump();
}

// ─── Visualization ────────────────────────────────────────────────────────────

void RLBridge::drawZoneCircles()
{
    cCanvas* canvas = getParentModule()->getCanvas();
    for (auto* f : zoneFigures_) { canvas->removeFigure(f); delete f; }
    zoneFigures_.clear();

    auto addCircle = [&](double cx, double cy, double r,
                         const char* fillHex, const char* lineHex) {
        auto* oval = new cOvalFigure();
        oval->setBounds(cFigure::Rectangle(cx - r, cy - r, 2.0 * r, 2.0 * r));
        oval->setFilled(true);
        oval->setFillColor(cFigure::Color(fillHex));
        oval->setFillOpacity(0.2);
        oval->setLineColor(cFigure::Color(lineHex));
        oval->setLineOpacity(0.5);
        oval->setLineWidth(1.5);
        oval->setZIndex(-1.0);
        canvas->addFigure(oval);
        zoneFigures_.push_back(oval);
    };
    for (auto& c : incidentZoneCenters_)
        addCircle(c.x, c.y, 200.0, "#FF4444", "#CC0000");
    addCircle(commandZoneCenter_.x, commandZoneCenter_.y, 100.0, "#4444FF", "#0000CC");
    addCircle(patientsWaitingCenter_.x, patientsWaitingCenter_.y, 150.0, "#FFAA22", "#CC7700");
    for (auto& c : ccsZoneCenters_)
        addCircle(c.x, c.y, 100.0, "#44BB44", "#009900");
}

void RLBridge::updateDronePairLines()
{
    static constexpr double RANGE = 1000.0;
    int lineIdx = 0;
    for (int i = 0; i < numDrones_; ++i) {
        for (int j = i + 1; j < numDrones_; ++j, ++lineIdx) {
            if (lineIdx >= (int)dronePairLines_.size()) break;
            auto* line = dronePairLines_[lineIdx];
            double ix, iy, jx, jy;
            getDronePosition(i, ix, iy); getDronePosition(j, jx, jy);
            double dx = ix - jx, dy = iy - jy;
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist <= RANGE) {
                line->setStart(cFigure::Point(ix, iy));
                line->setEnd(cFigure::Point(jx, jy));
                line->setVisible(true);
            } else { line->setVisible(false); }
        }
    }
}

// ─── Signal listener ──────────────────────────────────────────────────────────


// ── Track-agnostic transmitter-address resolution ────────────────────────────
// The OMNET track runs Ieee80211Mac; the OM2G track runs Ieee802154Mac.  The
// three call sites below all want the same thing -- "which node sent this
// frame?" -- off a header that has not yet been decapsulated, so they cannot
// use MacAddressInd.  Peeking the wrong header type is a HARD ERROR in INET
// ("Cannot convert chunk from type inet::Ieee802154MacHeader to type
// inet::ieee80211::Ieee80211MacHeader"), which is exactly how the OM2G track
// died on its first run.  PF_ALLOW_NULLPTR makes a type mismatch return null
// instead of throwing, so this probes 802.11 first and falls back to 802.15.4.
//
// Returns UNSPECIFIED_ADDRESS when neither matches (e.g. an 802.11 ACK/CTS,
// which is a one-address frame and carries no transmitter address at all).
static MacAddress om2gResolveTransmitterAddress(const Packet* pkt)
{
    if (!pkt) return MacAddress::UNSPECIFIED_ADDRESS;
    // Peek the GENERIC base chunk and dynamic-cast, never peek<ConcreteHeader>.
    // peek<T> on a mismatched type routes through Chunk::convertChunk, which
    // throws "Cannot convert chunk from type A to type B" -- and
    // PF_ALLOW_NULLPTR does NOT suppress that (it only covers a genuinely
    // empty result; see Chunk::checkPeekResult). Peeking <Chunk> always
    // matches, so the cast below is a pure type TEST with no conversion.
    const auto& front = pkt->peekAtFront<Chunk>(b(-1), Chunk::PF_ALLOW_NULLPTR
                                                     | Chunk::PF_ALLOW_EMPTY
                                                     | Chunk::PF_ALLOW_INCOMPLETE
                                                     | Chunk::PF_ALLOW_INCORRECT
                                                     | Chunk::PF_ALLOW_IMPROPERLY_REPRESENTED);
    if (front == nullptr) return MacAddress::UNSPECIFIED_ADDRESS;
    if (auto twoAddr = dynamicPtrCast<const ieee80211::Ieee80211TwoAddressHeader>(front))
        return twoAddr->getTransmitterAddress();   // OMNET track
    if (auto h154 = dynamicPtrCast<const Ieee802154MacHeader>(front))
        return h154->getSrcAddr();                 // OM2G track
    return MacAddress::UNSPECIFIED_ADDRESS;
}

void RLBridge::receiveSignal(cComponent* src, simsignal_t id,
                             cObject* obj, cObject* details)
{
    if (id == receptionEndedSignal) {
        // Diagnostic (see DiagCounters above): item (5), independent
        // of the src->getName()=="radio" branch below and of packetSentToUpper
        // firing at all.
        cModule* container = src->getParentModule();
        while (container && std::string(container->getName()) != "host")
            container = container->getParentModule();
        if (container) {
            int idx = container->getIndex();
            if (idx >= 0 && idx < numDrones_)
                g_diag[idx].receptionEndedTotal++;
        }
        return;
    }

    if (id == transmissionStartedSignal || id == transmissionEndedSignal) {
        // Diagnostic: unconditional TX-side signals (Radio.cc emits
        // both regardless of what triggered the transmission — app data, AODV
        // control, ARP, or 802.11 ACK/management frames) — independent of the
        // packetSentToUpper src-name branch below.
        cModule* container = src->getParentModule();
        while (container && std::string(container->getName()) != "host"
                         && std::string(container->getName()) != "user")
            container = container->getParentModule();
        if (container) {
            std::string cname = container->getName();
            int idx = container->getIndex();
            if (cname == "host" && idx >= 0 && idx < numDrones_) {
                if (id == transmissionStartedSignal) g_diag[idx].transmissionStarted++;
                else                                  g_diag[idx].transmissionEnded++;
            } else if (cname == "user" && idx >= 0 && idx < numUsers_) {
                if (id == transmissionStartedSignal) g_diagUser[idx].transmissionStarted++;
                else                                  g_diagUser[idx].transmissionEnded++;
            }
        }
        return;
    }

    if (id == packetDroppedSig_ || id == linkBrokenSig_) {
        // DROP_ANY printf gated: at high drop rates this printf+fflush is itself
        // enormous syscall overhead AND it scales with the failure rate, so disease
        // and instrumentation become indistinguishable. Set OMNET_DROP_DIAG=1 to
        // re-enable. The drop-column bookkeeping below is NEVER gated.
        static const bool dropDiag_ = (std::getenv("OMNET_DROP_DIAG") != nullptr);
        if (dropDiag_ && id == packetDroppedSig_) {
            auto* dd = dynamic_cast<PacketDropDetails*>(details);
            std::string dstStr = "?";
            if (auto* pk = dynamic_cast<Packet*>(obj)) {
                auto ind = pk->findTag<L3AddressInd>();
                if (ind) dstStr = ind->getDestAddress().str();
            }
            std::printf("DROP_ANY:%.3f %s reason=%d ipdst=%s\n",
                        simTime().dbl(),
                        src->getFullPath().c_str(),
                        dd ? (int)dd->getReason() : -1,
                        dstStr.c_str());
            std::fflush(stdout);
        }
        cModule* container = src->getParentModule();
        bool isNetLayer = false;
        while (container && std::string(container->getName()) != "host") {
            if (std::string(container->getName()) == "ipv4") isNetLayer = true;
            container = container->getParentModule();
        }
        if (!container) return;
        int idx = container->getIndex();
        if (idx < 0 || idx >= numDrones_) return;
        auto& counts = droneEventCounts_[idx];
        if (id == linkBrokenSig_) {
            counts.linkBroken++;
            return;
        }
        auto* dropDetails = dynamic_cast<PacketDropDetails*>(details);
        if (!dropDetails) return;
        if (isNetLayer) {
            switch (dropDetails->getReason()) {
                case NO_ROUTE_FOUND:            counts.netDropNoRoute++;       break;
                case ADDRESS_RESOLUTION_FAILED: counts.netDropAddrResFailed++; break;
                case HOP_LIMIT_REACHED:         counts.netDropHopLimit++;      break;
                case FORWARDING_DISABLED:       counts.netDropFwdDisabled++;   break;
                case NO_INTERFACE_FOUND:        counts.netDropNoIface++;       break;
                case LIFETIME_EXPIRED:          counts.netDropLifetime++;      break;
                case INCORRECTLY_RECEIVED:      counts.netDropChecksum++;      break;
                default:                        counts.netDropUndefined++;     break;
            }
            return;
        }
        PacketDropReason reason = dropDetails->getReason();
        switch (reason) {
            case RETRY_LIMIT_REACHED:  counts.dropRetry++;    break;
            case QUEUE_OVERFLOW:       counts.dropQueue++;    break;
            case INCORRECTLY_RECEIVED: counts.dropCorrupt++;  break;
            case NOT_ADDRESSED_TO_US:  counts.dropNotForUs++; break;
            default: break;
        }

        // ── Raw per-event record for the 4 MAC-layer drop reasons the event
        // token tracks (see RLBridge.h EventRecord).
        if (reason == RETRY_LIMIT_REACHED || reason == QUEUE_OVERFLOW ||
            reason == INCORRECTLY_RECEIVED || reason == NOT_ADDRESSED_TO_US)
        {
            auto* pkt = dynamic_cast<Packet*>(obj);
            EventRecord rec;
            rec.isDrop       = 1.0;
            rec.dropRetry    = (reason == RETRY_LIMIT_REACHED)  ? 1.0 : 0.0;
            rec.dropQueue    = (reason == QUEUE_OVERFLOW)       ? 1.0 : 0.0;
            rec.dropCorrupt  = (reason == INCORRECTLY_RECEIVED) ? 1.0 : 0.0;
            rec.dropNotForUs = (reason == NOT_ADDRESSED_TO_US)  ? 1.0 : 0.0;

            if (pkt) {
                rec.size = pkt->getByteLength();

                if (reason == RETRY_LIMIT_REACHED || reason == QUEUE_OVERFLOW) {
                    // Both fire on OUR OWN outgoing packet (ACK never came
                    // back / local TX queue full) — nothing was received, so
                    // there is no SNIR/PER/BER to report and no remote
                    // sender: this drone is itself the origin.
                    rec.fromDrone = 1.0;
                } else {
                    // INCORRECTLY_RECEIVED (Rx, pre-MAC-decapsulation) and
                    // NOT_ADDRESSED_TO_US (Ds, post-MAC-decapsulation) both
                    // fire on a frame that WAS actually received from PHY, so
                    // SnirInd/ErrorRateInd (attached back at Radio::endReception)
                    // are still present in both cases — verified against INET
                    // 4.6 Rx.cc/Ds.cc/Ieee80211Mac.cc: nothing strips them.
                    if (auto snirTag = pkt->findTag<SnirInd>()) {
                        double snirLin = snirTag->getMinimumSnir();
                        if (!std::isnan(snirLin)) {
                            rec.snir = math::fraction2dB(snirLin);
                            rec.snirValid = 1.0;
                        }
                    }
                    if (auto errTag = pkt->findTag<ErrorRateInd>()) {
                        double perVal = errTag->getPacketErrorRate();
                        double berVal = errTag->getBitErrorRate();
                        if (!std::isnan(perVal)) rec.per = perVal;
                        if (!std::isnan(berVal)) rec.ber = berVal;
                    }

                    if (auto macTag = pkt->findTag<MacAddressInd>()) {
                        // NOT_ADDRESSED_TO_US: Ieee80211Mac::decapsulate()
                        // already ran (Ieee80211Mac::sendUpFrame calls it
                        // before dispatching to Ds), so MacAddressInd is
                        // already attached — use it directly.
                        auto it = macIsDrone_.find(macTag->getSrcAddress());
                        if (it != macIsDrone_.end())
                            rec.fromDrone = it->second ? 1.0 : 0.0;
                    } else {
                        // INCORRECTLY_RECEIVED: decapsulate() never ran (Rx
                        // drops before the frame reaches Ieee80211Mac), so no
                        // MacAddressInd yet — read the transmitter address
                        // directly off the still-intact MAC header instead.
                        // INET's FCS-failure model only flags
                        // hasBitError()/isCorrect()=false; it never mutates
                        // header field bytes (verified against
                        // Rx::isFcsOk()), so the header content — including
                        // the transmitter address — is still genuine even for
                        // a corrupted frame. dynamicPtrCast returns null for
                        // one-address frames (ACK/CTS have no transmitter
                        // address field), left unclassified in that case.
                        MacAddress txAddr = om2gResolveTransmitterAddress(pkt);
                        if (!txAddr.isUnspecified()) {
                            auto it2 = macIsDrone_.find(txAddr);
                            if (it2 != macIsDrone_.end())
                                rec.fromDrone = it2->second ? 1.0 : 0.0;
                        }
                    }
                }
            }
            recordEvent(idx, rec);
        }
        return;
    }

    if (id != packetReceivedSig_ && id != packetSentToUpperSig_) return;
    auto* pkt = dynamic_cast<cPacket*>(obj);
    if (!pkt) return;
    cModule* container = src->getParentModule();
    while (container && std::string(container->getName()) != "host"
                     && std::string(container->getName()) != "user")
        container = container->getParentModule();
    if (!container) return;
    std::string name = container->getName();
    int idx = container->getIndex();
    if (id == packetReceivedSig_) {
        if (name == "host" && idx >= 0 && idx < numDrones_)
            droneBytesRx_[idx] += pkt->getByteLength();
        else if (name == "user" && idx >= 0 && idx < numUsers_)
            userBytesRx_[idx] += pkt->getByteLength();
    } else if (id == packetSentToUpperSig_) {
        if (name == "host" && idx >= 0 && idx < numDrones_) {
            droneBytesUp_[idx] += pkt->getByteLength();
            // Diagnostic (see DiagCounters above): items (1a)/(1b).
            g_diag[idx].srcNameHits[src->getName()]++;
            // Diagnostic: Radio-layer packetSentToUpper == successful reception.
            if (std::string(src->getName()) == "radio") {
                g_diag[idx].radioBranchHits++;
                droneEventCounts_[idx].rx++;
                droneLastRxTime_[idx] = simTime();

                // ── Measured received power (RSSI), per-(observer, source) ──
                // Hooked at the RADIO's own packetSentToUpper (Radio::sendUp,
                // called unconditionally from Radio::endReception for every
                // completed reception regardless of outcome — see Radio.cc)
                // rather than the MAC's packetSentToUpper, because the MAC
                // only reaches sendUp() for frames that pass
                // Dcf::/Hcf::processLowerFrame's isForUs() check
                // (unicast-to-me/broadcast/multicast) — everything else is
                // dropped as NOT_ADDRESSED_TO_US before it ever reaches the
                // MAC-level packetSentToUpper, so that hook only ever sees
                // the ~2 peers a drone happens to be relaying for, not every
                // decoded frame. SignalPowerInd IS already attached at this
                // radio-level hook (ReceiverBase::computeReceptionResult,
                // called from Radio::endReception before decapsulate/sendUp).
                // MacAddressInd is NOT attached yet here (only
                // Ieee80211Mac::decapsulate() adds it, which runs later, only
                // for frames that already passed the address filter) — so
                // resolve the sender from the still-intact Ieee80211 MAC
                // header's transmitter address instead, same technique
                // already used below for the INCORRECTLY_RECEIVED drop path.
                if (auto* inetPkt = dynamic_cast<Packet*>(pkt)) {
                    if (auto powerTag = inetPkt->findTag<SignalPowerInd>()) {
                        g_diag[idx].powerTagYes++;  // Diagnostic: item (2)
                        double powerW = powerTag->getPower().get<W>();
                        if (!std::isnan(powerW)) {
                            MacAddress srcMac;
                            bool haveSrcMac = false;
                            if (auto macTag = inetPkt->findTag<MacAddressInd>()) {
                                srcMac = macTag->getSrcAddress();
                                haveSrcMac = true;
                                g_diag[idx].resolvedViaMacTag++;  // Diagnostic: item (3a)
                            } else {
                                MacAddress txAddr = om2gResolveTransmitterAddress(inetPkt);
                                if (!txAddr.isUnspecified()) {
                                    srcMac = txAddr;
                                    haveSrcMac = true;
                                    g_diag[idx].resolvedViaHeaderFallback++;  // Diagnostic: item (3b)
                                } else {
                                    g_diag[idx].resolveFailed++;  // Diagnostic: item (3c)
                                }
                            }
                            if (haveSrcMac) {
                                double dbm = math::mW2dBmW(powerW * 1000.0);
                                SimTime now = simTime();
                                auto dIt = macToDroneIdx_.find(srcMac);
                                if (dIt != macToDroneIdx_.end()) {
                                    int peerIdx = dIt->second;
                                    peerDroneRssiDbm_[idx][peerIdx]   = dbm;
                                    peerDroneRssiValid_[idx][peerIdx] = 1.0;
                                    peerDroneRssiTime_[idx][peerIdx]  = now;
                                    g_diag[idx].mappedToDrone++;  // Diagnostic: item (4)
                                } else {
                                    auto uIt = macToUserIdx_.find(srcMac);
                                    if (uIt != macToUserIdx_.end()) {
                                        int userIdx = uIt->second;
                                        userRssiDbm_[idx][userIdx]   = dbm;
                                        userRssiValid_[idx][userIdx] = 1.0;
                                        userRssiTime_[idx][userIdx]  = now;
                                        g_diag[idx].mappedToUser++;  // Diagnostic: item (4)
                                    } else {
                                        g_diag[idx].mappedToNeither++;  // Diagnostic: item (4)
                                    }
                                }
                            }
                        } else {
                            g_diag[idx].powerNan++;  // Diagnostic
                        }
                    } else {
                        g_diag[idx].powerTagNo++;  // Diagnostic: item (2)
                    }
                }
            }

            // Raw per-event record for a successful reception (is_drop=0).
            // Read at the MAC-level firing of this SAME signal (mac's own
            // packetSentToUpper, emitted right after the radio's, via
            // MacProtocolBase::sendUp) rather than the radio-level one above,
            // because only Ieee80211Mac::decapsulate() attaches MacAddressInd
            // — SnirInd/ErrorRateInd survive from the radio stage either way
            // (verified against INET 4.6 Radio.cc/Ieee80211Mac.cc: nothing
            // strips tags in between).
            if (std::string(src->getName()) == "mac") {
                if (auto* inetPkt = dynamic_cast<Packet*>(pkt)) {
                    EventRecord rec;
                    rec.size = inetPkt->getByteLength();
                    if (auto snirTag = inetPkt->findTag<SnirInd>()) {
                        double snirLin = snirTag->getMinimumSnir();
                        if (!std::isnan(snirLin)) {
                            rec.snir = math::fraction2dB(snirLin);
                            rec.snirValid = 1.0;
                        }
                    }
                    if (auto errTag = inetPkt->findTag<ErrorRateInd>()) {
                        double perVal = errTag->getPacketErrorRate();
                        double berVal = errTag->getBitErrorRate();
                        if (!std::isnan(perVal)) rec.per = perVal;
                        if (!std::isnan(berVal)) rec.ber = berVal;
                    }
                    if (auto macTag = inetPkt->findTag<MacAddressInd>()) {
                        auto it = macIsDrone_.find(macTag->getSrcAddress());
                        if (it != macIsDrone_.end())
                            rec.fromDrone = it->second ? 1.0 : 0.0;
                    }
                    recordEvent(idx, rec);
                }
            }
        }
        else if (name == "user" && idx >= 0 && idx < numUsers_) {
            userBytesUp_[idx] += pkt->getByteLength();
            // Diagnostic: does this user ever receive a
            // drone-sourced frame? Count-only — mirrors the drone-side
            // radio-branch resolution above, writes nothing to observation
            // state (users have no RSSI arrays in RLBridge).
            if (std::string(src->getName()) == "radio") {
                g_diagUser[idx].radioBranchHits++;
                if (auto* inetPkt = dynamic_cast<Packet*>(pkt)) {
                    MacAddress srcMac;
                    bool haveSrcMac = false;
                    if (auto powerTag = inetPkt->findTag<SignalPowerInd>()) {
                        double powerW = powerTag->getPower().get<W>();
                        if (!std::isnan(powerW)) {
                            if (auto macTag = inetPkt->findTag<MacAddressInd>()) {
                                srcMac = macTag->getSrcAddress();
                                haveSrcMac = true;
                            } else {
                                MacAddress txAddr = om2gResolveTransmitterAddress(inetPkt);
                                if (!txAddr.isUnspecified()) {
                                    srcMac = txAddr;
                                    haveSrcMac = true;
                                }
                            }
                        }
                    }
                    if (haveSrcMac) {
                        auto it = macIsDrone_.find(srcMac);
                        if (it != macIsDrone_.end() && it->second)
                            g_diagUser[idx].fromDrone++;
                        else if (it != macIsDrone_.end())
                            g_diagUser[idx].fromUser++;
                        else
                            g_diagUser[idx].unresolved++;
                    } else {
                        g_diagUser[idx].unresolved++;
                    }
                }
            }
        }
    }
}

// channel_busy_fraction: IRadio::receptionStateChangedSignal is emitted as an
// enum (-> intval_t), so it arrives through this cListener overload, not the
// cObject* one above.
void RLBridge::receiveSignal(cComponent* src, simsignal_t id,
                             intval_t value, cObject* details)
{
    if (id != physicallayer::IRadio::receptionStateChangedSignal) return;

    cModule* container = src->getParentModule();
    while (container && std::string(container->getName()) != "host")
        container = container->getParentModule();
    if (!container) return;
    int idx = container->getIndex();
    if (idx < 0 || idx >= numDrones_) return;

    SimTime now = simTime();
    bool wasBusy = (droneReceptionState_[idx] == physicallayer::IRadio::RECEPTION_STATE_BUSY ||
                    droneReceptionState_[idx] == physicallayer::IRadio::RECEPTION_STATE_RECEIVING);
    if (wasBusy)
        droneBusyTime_[idx] += (now - droneLastStateChangeTime_[idx]).dbl();

    droneReceptionState_[idx]      = (int)value;
    droneLastStateChangeTime_[idx] = now;
}

// ─── Action parsing ───────────────────────────────────────────────────────────

void RLBridge::parseActionJson(const std::string& s)
{
    json j = json::parse(s);
    if (!j.contains("actions"))
        throw cRuntimeError("RLBridge: action JSON missing 'actions'");
    auto actions = j["actions"];
    if (!actions.is_array() || (int)actions.size() != numDrones_)
        throw cRuntimeError("RLBridge: expected %d actions, got %d",
                            numDrones_, (int)actions.size());
    for (int i = 0; i < numDrones_; ++i) {
        auto a = actions[i];
        // 2 elements = movement only (every OMNET-track and un-gated OM2G
        // run); 3 = movement + the OM2G learned send-gate's 3rd action dim
        // (see om2g_env.py's OM2G_LEARNED_GATE). Nothing else is valid.
        if (!a.is_array() || (a.size() != 2 && a.size() != 3))
            throw cRuntimeError("RLBridge: action %d malformed", i);
        actionBuffer_[i] = {a[0].get<double>(), a[1].get<double>()};
        if (a.size() == 3)
            sendGateBuffer_[i] = a[2].get<double>();
    }
}

// ─── Routing table dump ───────────────────────────────────────────────────────

void RLBridge::dumpAllRoutingTables()
{
    cModule* net = getParentModule();
    std::cout << "===== ROUTING TABLE DUMP =====" << std::endl;
    auto dumpOne = [&](cModule* host, const char* label) {
        if (!host) return;
        auto* rt = dynamic_cast<IIpv4RoutingTable*>(host->getModuleByPath(".ipv4.routingTable"));
        if (!rt) return;
        std::cout << "--- " << label << " ---" << std::endl;
        for (int i = 0; i < rt->getNumRoutes(); ++i) {
            auto* r = rt->getRoute(i);
            std::cout << "  " << r->getDestination() << "/" << r->getNetmask().getNetmaskLength()
                      << " dev " << (r->getInterface() ? r->getInterface()->getInterfaceName() : "?")
                      << std::endl;
        }
        if (rt->getNumRoutes() == 0) std::cout << "  (empty)" << std::endl;
    };
    for (int i = 0; i < numDrones_; ++i) {
        char l[32]; snprintf(l, sizeof(l), "host[%d]", i);
        dumpOne(net->getSubmodule("host", i), l);
    }
    for (int i = 0; i < numUsers_; ++i) {
        char l[32]; snprintf(l, sizeof(l), "user[%d]", i);
        dumpOne(net->getSubmodule("user", i), l);
    }
    std::cout << "============================" << std::endl;
}

// ─── Finish ───────────────────────────────────────────────────────────────────

void RLBridge::finish()
{
    // Diagnostic dump (see DiagCounters above) — one line per drone,
    // printed once at episode end. Count-only; does not affect behavior.
    for (int i = 0; i < numDrones_; ++i) {
        auto& d = g_diag[i];
        std::ostringstream names;
        for (auto& kv : d.srcNameHits)
            names << kv.first << ":" << kv.second << ",";
        std::printf("[DIAG_RSSI] drone=%d radio_hits=%ld src_names=[%s] "
                    "power_tag_yes=%ld power_tag_no=%ld power_nan=%ld "
                    "resolved_mac_tag=%ld resolved_header_fallback=%ld resolve_failed=%ld "
                    "mapped_drone=%ld mapped_user=%ld mapped_neither=%ld "
                    "reception_ended_total=%ld "
                    "tx_started=%ld tx_ended=%ld\n",
                    i, d.radioBranchHits, names.str().c_str(),
                    d.powerTagYes, d.powerTagNo, d.powerNan,
                    d.resolvedViaMacTag, d.resolvedViaHeaderFallback, d.resolveFailed,
                    d.mappedToDrone, d.mappedToUser, d.mappedToNeither,
                    d.receptionEndedTotal,
                    d.transmissionStarted, d.transmissionEnded);
    }
    // Diagnostic: user-side reception
    // attribution (does a user ever receive a drone-sourced frame?) + user TX.
    for (int i = 0; i < numUsers_; ++i) {
        auto& u = g_diagUser[i];
        std::printf("[DIAG_USER] user=%d radio_hits=%ld from_drone=%ld from_user=%ld "
                    "unresolved=%ld tx_started=%ld tx_ended=%ld\n",
                    i, u.radioBranchHits, u.fromDrone, u.fromUser, u.unresolved,
                    u.transmissionStarted, u.transmissionEnded);
    }
    std::fflush(stdout);

    if (par("dumpRoutes").boolValue()) dumpAllRoutingTables();
    try {
        json j;
        j["type"]                  = "obs";
        j["step"]                  = stepCounter_;
        j["done"]                  = true;
        j["positions"]             = json::array();
        j["throughput"]            = json::array();
        j["user_positions"]        = json::array();
        j["user_area"]             = json::array();
        j["sender_pairs"]          = json::array();
        j["active_broadcasts"]     = json::array();
        j["talk_group_heads"]      = json::array();
        j["ccs_zone_centers"]      = json::array();
        j["user_goodput"]          = json::array();
        j["drone_user_counts"]     = json::array();
        j["noise_floor"]           = json::array();
        j["zone_centers"]          = json::object();
        j["num_incidents"]         = 0;
        j["talk_group_memberships"]= json::array();
        j["events"]                 = json::array();
        j["peer_rssi"]               = json::array();
        j["peer_rssi_valid"]         = json::array();
        j["peer_rssi_age"]           = json::array();
        j["user_rssi"]               = json::array();
        j["user_rssi_valid"]         = json::array();
        j["user_rssi_age"]           = json::array();
        j["channel_busy"]           = json::array();
        j["time_since_rx"]          = json::array();
        std::string s = j.dump();
        zmq::message_t m(s.data(), s.size());
        sock_->send(m, zmq::send_flags::dontwait);
    } catch (...) {}
}

} // namespace inet
