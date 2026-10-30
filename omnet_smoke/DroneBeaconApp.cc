//
// DroneBeaconApp.cc
//
// Periodic hello beacon over UDP multicast. See DroneBeaconApp.ned for why
// multicast rather than broadcast.
//
#include "DroneBeaconApp.h"

#include <cstdlib>
#include <cstring>
#include <vector>
#include "RLBridge.h"

#include <cmath>

#include "inet/common/ModuleAccess.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/chunk/BytesChunk.h"
#include "inet/networklayer/common/NetworkInterface.h"
#include "inet/networklayer/contract/IInterfaceTable.h"
#include "inet/networklayer/ipv4/IIpv4RoutingTable.h"
#include "inet/networklayer/ipv4/Ipv4Route.h"

namespace inet {

Define_Module(DroneBeaconApp);

DroneBeaconApp::~DroneBeaconApp()
{
    cancelAndDelete(beaconTimer_);
}

void DroneBeaconApp::initialize(int stage)
{
    cSimpleModule::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        droneId_     = par("droneId");
        interval_    = par("beaconInterval");
        beaconPort_  = par("beaconPort");
        beaconBytes_ = par("beaconBytes");
        dynamicBeaconSize_  = par("dynamicBeaconSize");
        beaconFixedBytes_   = par("beaconFixedBytes");
        beaconBytesPerUser_ = par("beaconBytesPerUser");
        sendOnDeltaEnabled_       = par("sendOnDeltaEnabled");
        sendOnDeltaPosThresholdM_ = par("sendOnDeltaPosThresholdM");
        learnedGateEnabled_       = par("learnedGateEnabled");
        beaconGroup_ = Ipv4Address(par("beaconGroup").stringValue());

        const char* bridgePath = par("bridgeModule").stringValue();
        cModule* m = getModuleByPath(bridgePath);
        if (!m)
            throw cRuntimeError("DroneBeaconApp[%d]: bridge not found at '%s'",
                                droneId_, bridgePath);
        bridge_ = check_and_cast<RLBridge*>(m);

        beaconTimer_ = new cMessage("beaconTimer");
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        // Strip auto-installed wlan0 route
        {
            cModule* host = getContainingNode(this);
            auto* rt = check_and_cast<IIpv4RoutingTable*>(
                host->getSubmodule("ipv4")->getSubmodule("routingTable"));
            auto* ift = check_and_cast<IInterfaceTable*>(
                host->getSubmodule("interfaceTable"));
            NetworkInterface* stripWlan = ift->findInterfaceByName("wlan0");

            if (stripWlan) {
                for (int i = 0; i < rt->getNumRoutes(); ) {
                    Ipv4Route* r = rt->getRoute(i);
                    if (r->getSourceType() == IRoute::IFACENETMASK &&
                        r->getInterface() == stripWlan) {
                        rt->deleteRoute(r);
                    }
                    else {
                        ++i;
                    }
                }
            }
        }

        socket_.setOutputGate(gate("socketOut"));
        socket_.setCallback(this);
        socket_.bind(beaconPort_);
        socket_.joinMulticastGroup(beaconGroup_);

        simtime_t jitter = uniform(0, interval_.dbl());
        scheduleAt(simTime() + jitter + interval_, beaconTimer_);
    }
}

void DroneBeaconApp::handleMessage(cMessage* msg)
{
    if (msg == beaconTimer_) {
        // Opt-in cheap send-gating heuristic: if nothing worth reporting has
        // changed since the last beacon THIS drone actually sent, skip this
        // one entirely -- no packet, no airtime charge. "Changed" = moved
        // more than sendOnDeltaPosThresholdM_ OR the sensed-user count
        // differs (a coarse but cheap proxy for "my state report would look
        // different"). The very first beacon always sends (everSent_==false).
        // Learned gate takes priority over the fixed sendOnDelta rule when
        // both happen to be enabled (they shouldn't be, together, but this
        // keeps the precedence unambiguous rather than undefined).
        if (learnedGateEnabled_ && !bridge_->getLearnedGateWantsSend(droneId_)) {
            scheduleAt(simTime() + interval_, beaconTimer_);
            return;
        }
        if (!learnedGateEnabled_ && sendOnDeltaEnabled_ && everSent_) {
            double x = 0.0, y = 0.0;
            bridge_->getDronePosition(droneId_, x, y);
            double dx = x - lastSentX_, dy = y - lastSentY_;
            double moved = std::sqrt(dx * dx + dy * dy);
            int sensedNow = bridge_->getSensedUserCount(droneId_);
            if (moved < sendOnDeltaPosThresholdM_ && sensedNow == lastSentSensedCount_) {
                scheduleAt(simTime() + interval_, beaconTimer_);
                return;
            }
        }

        // OM2G: the payload is `beaconBytes` bytes of opaque state -- on the
        // OM2G track that number is the SIZE OF THE REAL STATE MESSAGE for the
        // configured arm (om2g_track.om2g_message_bytes), so the datagram this
        // app puts on the medium costs exactly the airtime the arm's message
        // costs.  The first 4 bytes carry the sender's droneId so the receiver
        // can attribute the message; the rest is filler, because the numeric
        // content rides the ZMQ channel while its COST rides the radio.
        //
        // A minimum of 4 bytes is enforced: below that the id does not fit and
        // delivery could not be attributed, which would disable the whole
        // staleness mechanism.
        //
        // dynamicBeaconSize_ (opt-in, default off): instead of the static
        // beaconBytes_ (padded to OM2G_MAX_USERS_CAP for the uncompressed
        // arm), size the payload to THIS drone's live sensed-user count --
        // an unseen user costs nothing on the wire, it is simply absent, not
        // padded. The model side is untouched: the receiver/env still
        // reconstructs a full padded tensor for the encoder from whichever
        // rows arrive (see om2g_track.py's OM2G_MSG_FIXED_FLOATS /
        // OM2G_OBS_FEATURES, which beaconFixedBytes_/beaconBytesPerUser_
        // mirror -- keep them in lockstep, they are passed from the same
        // Python source of truth via om2g_extra_args).
        int numBytes;
        if (dynamicBeaconSize_) {
            int sensed = bridge_->getSensedUserCount(droneId_);
            numBytes = beaconFixedBytes_ + sensed * beaconBytesPerUser_;
        }
        else {
            numBytes = beaconBytes_;
        }
        numBytes = numBytes < 4 ? 4 : numBytes;

        // ── MULTI-HOP: re-advertise every peer state we hold ────────────────
        // $OM2G_FORWARD_HOPS>0 turns the beacon from a one-hop hello into
        // link-state flooding, so information reaches the whole CONNECTED
        // COMPONENT instead of only direct neighbours. Single-hop measured
        // 81% of peer slots blank while the control plane used 2.2% of the
        // channel: the channel was empty and the information still could not
        // get there.
        //
        // Each forwarded entry costs a FULL state message on the wire
        // (`numBytes` each), because that is what re-transmitting somebody's
        // state actually costs. That is the trade-off under study: flooding
        // uncompressed state is unaffordable, flooding a compressed latent is not.
        static const int fwdHops = [] {
            const char* e = std::getenv("OM2G_FORWARD_HOPS");
            return e ? std::atoi(e) : 0;
        }();
        std::vector<int> fwdIds;
        std::vector<double> fwdTimes;
        if (fwdHops > 0 && bridge_ != nullptr) {
            bridge_->getKnownPeerStates(droneId_, fwdIds, fwdTimes);
            // header: 1 byte count, then 1 byte id + 8 bytes origin time each
            numBytes += (int)fwdIds.size() * numBytes;
        }

        std::vector<uint8_t> payload(numBytes, 0xBE);
        uint32_t idLE = (uint32_t)droneId_;
        payload[0] = (uint8_t)( idLE        & 0xFF);
        payload[1] = (uint8_t)((idLE >>  8) & 0xFF);
        payload[2] = (uint8_t)((idLE >> 16) & 0xFF);
        payload[3] = (uint8_t)((idLE >> 24) & 0xFF);
        // Forwarded table, only if it fits: count at [4], then (id, origin)
        // pairs of 1 + 8 bytes. It always fits, because every entry already
        // bought itself a whole message worth of payload above.
        if (fwdHops > 0 && !fwdIds.empty() && numBytes >= 5) {
            int maxEntries = (numBytes - 5) / 9;
            int nEnt = (int)fwdIds.size() < maxEntries ? (int)fwdIds.size() : maxEntries;
            payload[4] = (uint8_t)nEnt;
            for (int k = 0; k < nEnt; ++k) {
                int off = 5 + k * 9;
                payload[off] = (uint8_t)fwdIds[k];
                double t = fwdTimes[k];
                uint8_t buf[8];
                std::memcpy(buf, &t, 8);
                for (int b = 0; b < 8; ++b) payload[off + 1 + b] = buf[b];
            }
        }

        auto chunk = makeShared<BytesChunk>();
        chunk->setBytes(payload);

        auto* pkt = new Packet("droneBeacon");
        pkt->insertAtBack(chunk);

        socket_.sendTo(pkt, beaconGroup_, beaconPort_);
        bridge_->noteBeaconSent(droneId_, numBytes);

        if (sendOnDeltaEnabled_) {
            bridge_->getDronePosition(droneId_, lastSentX_, lastSentY_);
            lastSentSensedCount_ = bridge_->getSensedUserCount(droneId_);
            everSent_ = true;
        }

        EV_DETAIL << "DroneBeaconApp[" << droneId_
                  << "]: beacon sent to " << beaconGroup_
                  << ", user_count=" << bridge_->getUserCount(droneId_)
                  << endl;

        scheduleAt(simTime() + interval_, beaconTimer_);
    }
    else {
        socket_.processMessage(msg);
    }
}

void DroneBeaconApp::socketDataArrived(UdpSocket*, Packet* pkt)
{
    // OM2G: reaching here means the datagram was fully reassembled and
    // delivered -- every IPv4 fragment survived.  A 12-fragment uncompressed
    // message therefore arrives strictly less often than a 1-frame compressed
    // one, and the difference is measured, not assumed.
    int senderId = -1;
    int payloadBytes = (int)pkt->getByteLength();
    try {
        auto bytes = pkt->peekDataAsBytes();
        const auto& v = bytes->getBytes();
        if (v.size() >= 4) {
            senderId = (int)((uint32_t)v[0]
                           | ((uint32_t)v[1] <<  8)
                           | ((uint32_t)v[2] << 16)
                           | ((uint32_t)v[3] << 24));
        }
    }
    catch (const cRuntimeError&) {
        senderId = -1;   // malformed/short payload: count the bytes, no attribution
    }
    bridge_->noteBeaconReceived(droneId_, senderId, payloadBytes);

    // Multi-hop: adopt every peer state the sender relayed, carrying the
    // ORIGIN timestamp so a relayed copy can never look fresher than it is.
    // noteForwardedPeerState() keeps the newer of the two, so flooding
    // converges rather than oscillating.
    try {
        auto bytes = pkt->peekDataAsBytes();
        const auto& v = bytes->getBytes();
        if (v.size() >= 5) {
            int nEnt = (int)v[4];
            for (int k = 0; k < nEnt; ++k) {
                size_t off = 5 + (size_t)k * 9;
                if (off + 9 > v.size()) break;
                int aboutId = (int)v[off];
                double origin = 0.0;
                uint8_t buf[8];
                for (int b = 0; b < 8; ++b) buf[b] = v[off + 1 + b];
                std::memcpy(&origin, buf, 8);
                bridge_->noteForwardedPeerState(droneId_, aboutId, origin);
            }
        }
    }
    catch (const cRuntimeError&) { /* short/malformed: direct hop still counted */ }

    delete pkt;
}

void DroneBeaconApp::finish()
{
    socket_.close();
}

} // namespace inet
