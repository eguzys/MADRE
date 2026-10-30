// RLBridge.h
#ifndef __INET_RLBRIDGE_H
#define __INET_RLBRIDGE_H

#include <climits>
#include <random>
#include <string>
#include <vector>
#include <memory>

#include "inet/common/INETDefs.h"
#include "inet/common/geometry/common/Coord.h"
#include "inet/linklayer/common/MacAddress.h"
#include "inet/networklayer/contract/ipv4/Ipv4Address.h"
#include <omnetpp/clistener.h>

#include <map>

namespace zmq { class context_t; class socket_t; }
namespace omnetpp { class cLineFigure; class cOvalFigure; }

namespace inet {

class UserBonnTrafficApp;

// ─── TrafficParams: populated from UserBonnTrafficApp NED params ──────────────
// (Aschenbruck IPCCC 2007 + LCN 2006 constants)
struct TrafficParams {
    double headInitiatorProb    = 0.46;
    double meshProb             = 0.136;
    double transHeadToHead      = 0.142;
    double transMemberToMember  = 0.190;
    double callContinuationProb = 0.7;
    double callCHT_meanlog      = 0.4438584;
    double callCHT_sdlog        = 0.7766512;
    double callIdle_shape       = 1.530391;
    double callIdle_rate        = 1.606532;
    double convIdle_meanlog     = 1.390844;
    double convIdle_sdlog       = 1.628825;
    double convIdle_shift       = 3.0;
    double melpeIntervalMs      = 67.5;
    bool   paramsLoaded         = false;
};

// ─── TalkGroupProcess: per-talk-group conversation-process state ──────────────
struct TalkGroupProcess {
    int id   = -1;
    int head = -1;
    std::vector<int> members;

    int  current   = -1;   // current speaker
    int  other     = -1;   // the other party
    int  initiator = -1;   // mesh-call initiator (plays H role)
    bool isMesh    = false;

    // PADRE traffic mode only: sender/partner are fixed for the whole episode
    // (set once in buildPadreSenders). current/other get reset to -1 by
    // onCallEnd when a conversation fully ends, so fixedPartner preserves the
    // partner across conversations without re-running pickPadrePartner.
    int  fixedPartner = -1;

    // $OM2G_DEMAND_RESHUFFLE_STEPS: which demand epoch fixedPartner was last
    // drawn for. -1 = never redrawn. See the conv-idle handler in RLBridge.cc.
    int  demandEpoch  = -1;

    enum class State { CONV_IDLE, IN_CALL, BETWEEN_CALLS } state = State::CONV_IDLE;

    cMessage* convIdleTimer  = nullptr;  // fires after T_conv_idle
    cMessage* callEndTimer   = nullptr;  // fires after T_call_cht
    cMessage* callStartTimer = nullptr;  // fires after T_call_idle (between calls)
    cMessage* packetTimer    = nullptr;  // repeating MELPE-interval timer
};

// ─── RLBridge ─────────────────────────────────────────────────────────────────

class INET_API RLBridge : public cSimpleModule, public cListener
{
  public:
    RLBridge();
    virtual ~RLBridge();

    void registerDrone(int droneId, cModule* mobilityModule);
    void getAction(int droneId, double& outDx, double& outDy) const;
    int  getUserCount(int droneId) const;
    // Live sensed-user count (sum of userRssiValid_[droneId][*]) -- distinct
    // from getUserCount(), which is the ASSOCIATION count. This is what a
    // drone would actually include a row for if its outgoing state message
    // carried only sensed users instead of padding to OM2G_MAX_USERS_CAP.
    int  getSensedUserCount(int droneId) const;
    // OM2G learned send-gate readout: true iff sendGateBuffer_[droneId] > 0
    // (or the buffer is empty/out of range, which defaults to "send").
    bool getLearnedGateWantsSend(int droneId) const;
    void getDronePosition(int droneId, double& outX, double& outY) const;
    // Single source of truth for step count (Task 3).
    int  getStepCounter() const { return stepCounter_; }
    void notifyAllDronesOfAssociation(int userId, Ipv4Address userIp,
                                      int newDroneId, int oldDroneId);

    // Per-episode layout queries.
    Coord getSampledUserPosition(int userId) const;
    int   getUserArea(int userId) const;   // 0=incident,1=patients-waiting,2=CCS,3=command
    int   getNumIncidents() const { return numSampledIncidents_; }

    // Called by UserBonnTrafficApp at INITSTAGE_APPLICATION_LAYER.
    void  registerTrafficApp(int userId, UserBonnTrafficApp* app, const TrafficParams& params);

    // ── OM2G: inter-agent state-message accounting ───────────────────────────
    // ⚠️ OM2G TRACK (IEEE 802.15.4 narrowband).  Additive and inert on the
    // OMNET track: DroneBeaconApp calls these unconditionally, but nothing in
    // the OMNET Python side reads the JSON keys they feed, and no OMNET
    // observation column changes.  See code/om2g/OM2G_README.md.
    //
    // These make the claim "the state message actually traverses the network"
    // MEASURED rather than asserted.  noteBeaconSent/Received are called by
    // DroneBeaconApp on the real UDP multicast path, so an undelivered or
    // fragment-lost message is simply never recorded and the observer's peer
    // row goes stale -- exactly as it would on a real radio.
    void noteBeaconSent(int txDroneId, int payloadBytes);
    void noteBeaconReceived(int rxDroneId, int txDroneId, int payloadBytes);

    // ── MULTI-HOP STATE DISSEMINATION ($OM2G_FORWARD_HOPS) ──────────────────
    // A single-hop beacon means a drone only ever learns about peers inside
    // its own radio range. Measured consequence: 81% of peer slots blank
    // (45.6 of 56), while the control plane used only 2.2% of the channel --
    // the channel was nearly EMPTY and the information still did not arrive.
    // Real link-state protocols (OLSR et al.) forward neighbour state; this
    // is that, with the airtime fully charged, so every forwarded entry
    // costs what it costs. `originTimeSec` is the time the ORIGINAL sender
    // emitted the state, not the forward time, so a relayed report keeps its
    // true age in the staleness features.
    void noteForwardedPeerState(int rxDroneId, int aboutDroneId,
                                double originTimeSec);
    // Peers this drone currently knows about, newest-first, for re-advertising.
    int  getKnownPeerStates(int droneId, std::vector<int>& ids,
                            std::vector<double>& originTimes) const;

    // Returns multicast address for talk group tgId (0-3 local, 4 global).
    Ipv4Address getTalkGroupMulticast(int tgId) const;

  protected:
    virtual void initialize(int stage) override;
    virtual int  numInitStages() const override { return NUM_INIT_STAGES; }
    virtual void handleMessage(cMessage* msg) override;
    virtual void finish() override;
    void dumpAllRoutingTables();

  private:
    std::string buildObservationJson(bool done);
    void        parseActionJson(const std::string& json);
    void        sampleEpisodeLayout(std::mt19937& rng);
    void        samplePadreClusters(std::mt19937& rng);
    std::string placementMode_;
    double      userSpread_ = 30.0;
    std::string trafficMode_;
    double      pCross_ = 1.0;
    double      senderProb_ = 0.2;
    int         pickPadrePartner(int sender);
    Coord       sampleInDisk(std::mt19937& rng, Coord center, double radius);
    void        buildTalkGroups();
    void        buildPadreSenders();
    void        drawZoneCircles();
    void        updateDronePairLines();

    // Conversation-process management.
    void startConversationProcesses();
    void onConvIdleExpired(int tgId);
    void startCall(int tgId);
    void onCallEnd(int tgId);
    void onCallStart(int tgId);
    void onPacketTick(int tgId);

    // RNG helpers (all use trafficRng_).
    double sampleConvIdle();
    double sampleCallCHT();
    double sampleCallIdle();
    double uniformTraffic();
    int    uniformIntTraffic(int lo, int hi);   // inclusive

    // ZMQ
    std::unique_ptr<zmq::context_t> ctx_;
    std::unique_ptr<zmq::socket_t>  sock_;
    std::string endpoint_;

    // Sim parameters
    SimTime syncInterval_;
    int numDrones_, numUsers_, maxSteps_;
    int stepCounter_ = 0;

    // $OM2G_USER_ROAM_M / $OM2G_USER_SPEED_MPS: users drift instead of standing
    // still. Their spawn point is remembered so the drift is a TETHERED random
    // walk -- cluster identity survives, only the exact positions go stale.
    std::vector<Coord> userHome_;

    cMessage* syncEvent_ = nullptr;

    // Drone state
    std::vector<cModule*>                  drones_;
    std::vector<std::pair<double,double>>  actionBuffer_;
    // OM2G learned send-gate (opt-in, OM2G_LEARNED_GATE=1): a 3rd,
    // policy-controlled value per drone in [-1,1], >0 meaning "send your
    // beacon this step". Defaults to 1.0 (always send) so it is a no-op
    // for every action message that doesn't carry a 3rd element -- i.e.
    // every existing OMNET-track and non-gated OM2G run.
    std::vector<double>                    sendGateBuffer_;
    std::vector<int>                       droneUserCounts_;

    // Throughput accounting
    std::vector<long> droneBytesRx_, droneBytesUp_;
    // OM2G control-plane byte counters, reset every RL step alongside
    // droneBytesRx_/droneBytesUp_ (buildObservationJson).  UDP-PAYLOAD bytes,
    // not on-air bytes -- the airtime figure is channel_busy, which the radio
    // measures directly.
    std::vector<long> ctrlBytesTx_, ctrlBytesRx_;
    std::vector<long> beaconTxCount_, beaconRxCount_;
    // [i][j] = sim time at which observer i last successfully received a state
    // message from peer j.  SimTime::ZERO = never (same convention as
    // peerDroneRssiTime_).  This is the ONLY thing that decides whether
    // observer i may use peer j's state: om2g_env.py serves the snapshot from
    // this timestamp, so a message that did not arrive is genuinely absent.
    std::vector<std::vector<SimTime>> beaconRxTime_;
    std::vector<std::vector<double>>  beaconRxValid_;
    int beaconPayloadBytes_ = 0;   // echoed into the obs JSON for provenance
    std::vector<long> userBytesRx_,  userBytesUp_;
    simsignal_t packetReceivedSig_   = -1;
    simsignal_t packetSentToUpperSig_= -1;

    // Association state
    std::vector<int>         userToDrone_;
    std::vector<Ipv4Address> userIps_;

    // ── Oracle routing (A2: hop-by-hop shortest paths over the drone graph) ──
    // Drones do not run a working discovery protocol against non-AODV user
    // leaves, so RLBridge installs the routes itself. This is oracle routing,
    // consistent with the position-oracle reward
    // (compute_position_oracle_reward).
    // DRONE_LINK_RANGE_M is the single source of truth for drone-drone
    // adjacency. It is NOT the canvas RANGE in updateDronePairLines() (that is
    // cosmetic). ~525 m @ 12 Mbps, 200 mW -- see omnetpp.ini.
    static constexpr double DRONE_LINK_RANGE_M = 525.0;

    // POS_DIAG collapse detector: previous sampled positions, so the
    // common-mode coherence can be formed over the 25-step window.
    std::vector<double> posPrevX_, posPrevY_;

    // GRAPH_DIAG state (filled by rebuildOracleRoutes, printed in handleMessage)
    int graphComps_   = 0;   // connected components of the drone adjacency graph
    int graphLargest_ = 0;   // size of the largest component
    int graphUnreach_ = 0;   // ordered drone pairs with no BFS path

    std::vector<Ipv4Address> droneIps_;          // droneIps_[d] = 10.0.0.(d+1)
    void resolveDroneIps();                      // once, at init
    void rebuildOracleRoutes();                  // full recompute, all drones

    // Drone spawn placement. Called from sampleEpisodeLayout AFTER the user
    // clusters exist, so the drones can be placed relative to real users.
    //   "ini"      - do nothing; keep initialX/initialY from omnetpp.ini (DEFAULT,
    //                so every pre-existing config is byte-identical)
    //   "cluster"  - shuffled round-robin over cluster centroids, Gaussian jitter.
    //                Every cluster gets a drone; every drone starts within r_u of
    //                real users, so reward (and therefore gradient) is nonzero at
    //                t=0. The drone graph MAY start fragmented -- learning to close
    //                it is the task.
    //   "staging"  - all drones on the LARGEST cluster's centroid, jittered.
    //                Starts fully connected but leaves other clusters over the
    //                horizon. Kept as an ablation.
    void sampleDroneSpawns(std::mt19937& rng);

    int         maxClusters_   = 3;          // see RLBridge.ned
    std::string droneSpawnMode_ = "ini";
    // Controlled-separation user spawn. "padre" = the production uniform-random
    // sampler; "dumbbell" = clusters on a line with separation pinned to
    // dumbbellD_, which removes the per-seed separation lottery.
    std::string userSpawnMode_ = "padre";
    double exploreP_ = 0.5;   // droneSpawnMode=="explore" mix
    double dumbbellD_ = 2445.0;
    double dumbbellJitter_ = 75.0;
    double maxNeighbourSep_ = 1750.0;  // upper edge of the separation band
    double      droneSpawnJitter_ = 50.0;    // Gaussian sigma, metres
    int  routeAuditReachable_ = -1;              // last audit: reachable pairs
    int  routeAuditTotal_     = -1;              // last audit: total pairs

    // Episode seed / arena
    int    episodeSeed_   = 0;
    double arenaWidth_    = 2000.0;
    double arenaHeight_   = 2000.0;
    double incidentZoneRadius_ = 300.0;
    double commandZoneRadius_  = 100.0;
    double ccsZoneRadius_      = 100.0;
    int    fieldPerIncidentMin_= 5;
    int    fieldPerIncidentMax_= 9;
    int    ccsStaffPerZoneMin_ = 2;
    int    ccsStaffPerZoneMax_ = 4;

    // Per-episode layout
    int              numSampledIncidents_ = 0;
    Coord            commandZoneCenter_;
    Coord            patientsWaitingCenter_;
    std::vector<Coord> ccsZoneCenters_;
    std::vector<Coord> incidentZoneCenters_;
    std::vector<Coord> sampledUserPositions_;
    std::vector<int>   sampledUserAreas_;          // 0-3 per user
    std::vector<int>   sampledUserIncidentIds_;    // kept for internal zone assignment only
    std::vector<std::string> sampledUserRolesInternal_; // used only for buildObservationJson zone_centers

    // Talk group state (Bonn model: 4 local + 1 global = 5 total)
    struct TalkGroup {
        std::string      type;      // "local_N" or "global"
        int              incident;  // area_id for local, -1 for global
        int              head;
        std::vector<int> members;
    };
    std::vector<TalkGroup>       talkGroups_;   // exactly 5
    std::vector<int>             talkGroupHeads_;  // int[5]
    std::vector<int>             activeSpeakers_;  // per-tg: -1=idle, else userId

    // Conversation processes
    std::vector<TalkGroupProcess>   tgProcs_;       // exactly 5
    std::vector<UserBonnTrafficApp*> trafficApps_;  // indexed by userId
    std::mt19937_64                  trafficRng_;
    TrafficParams                    trafficParams_;

    // Jammer interference tracking (Task 5)
    cModule* jammerModule_       = nullptr;  // nullptr if no jammer in network
    int      jammerActivateStep_ = INT_MAX;  // step at which jammer activates

    // ── Diagnostic MAC/PHY event-count measurement (no observation impact) ──
    // Per-drone counters of raw INET signals, reset every RL step and printed
    // as [EVENT_COUNT] lines. Used only to size a future per-event observation.
    struct EventCounts {
        long rx           = 0;  // successful receptions (Radio packetSentToUpper)
        long dropRetry    = 0;  // Mac: retry limit reached
        long dropQueue    = 0;  // Mac: queue overflow
        long dropCorrupt  = 0;  // Mac: incorrectly received
        long dropNotForUs = 0;  // Mac: not addressed to us
        long linkBroken   = 0;  // Mac: link broken

        // Network-layer (Ipv4, host[i].ipv4.ip) packetDropped breakdown.
        // Same signal name/id as the MAC counters above but subscribed on a
        // different module, so counted into separate fields below. AODV
        // itself (host[i].ipv4.ip's Aodv sibling) never emits packetDropped
        // (verified against inet/routing/aodv/Aodv.cc — it only subscribes
        // to linkBrokenSignal), so there is no separate AODV subscription.
        long netDropNoRoute       = 0;  // Ipv4: NO_ROUTE_FOUND
        long netDropAddrResFailed = 0;  // Ipv4: ADDRESS_RESOLUTION_FAILED
        long netDropHopLimit      = 0;  // Ipv4: HOP_LIMIT_REACHED
        long netDropFwdDisabled   = 0;  // Ipv4: FORWARDING_DISABLED
        long netDropNoIface       = 0;  // Ipv4: NO_INTERFACE_FOUND
        long netDropLifetime      = 0;  // Ipv4: LIFETIME_EXPIRED
        long netDropChecksum      = 0;  // Ipv4: INCORRECTLY_RECEIVED (IP header checksum)
        long netDropUndefined     = 0;  // Ipv4: reason left unset (OTHER_PACKET_DROP)

        long eventOverflow = 0;  // events dropped this step because the
                                  // per-drone EventRecord vector hit
                                  // MAX_EVENTS_CAP (see below)
    };
    std::vector<EventCounts> droneEventCounts_;
    simsignal_t packetDroppedSig_ = -1;
    simsignal_t linkBrokenSig_    = -1;
    void printAndResetEventCounts();

    // ── Raw per-event MAC/PHY observation set ───────────────────────────────
    // One EventRecord per individual received-or-dropped 802.11 frame this
    // drone's own MAC/radio observes, accumulated per drone and reset every
    // RL step — parallel to EventCounts above (which stays as the cooked
    // diagnostic sum; this is the uncompressed, individual event). Column
    // order is contractual: Python's build_raw_event_tokens() in
    // omnet_module/omnet_env.py depends on it exactly.
    //
    // CAP JUSTIFICATION — read this before changing MAX_EVENTS_CAP.
    // Measured on the frozen arena with drones transmitting: the RAW stream is ~480
    // events/drone/step, p99 1022, peak ~1482 — cap 32 saturates on 93% of
    // rows. ~87% of raw events are NOT_ADDRESSED_TO_US.
    // The CUT set (receptions + real-failure drops, not-for-us excluded) is
    // mean 4.09/drone/step, p99 30 — which is what 32 actually holds. So the
    // cap is only safe WITH the cut applied at recordEvent() below. If you
    // set OMNET_KEEP_NOTFORUS=1, raise this to >=1024 or the truncation
    // silently drops real receptions in arrival order.
    // Only the 4 MAC-layer drop reasons already tracked in EventCounts above
    // (dropRetry/dropQueue/dropCorrupt/dropNotForUs) are represented here;
    // linkBroken and IP-layer drops are not per-packet reception events with
    // SNIR/PER/BER and are out of scope for this raw MAC/PHY event set.
    static constexpr int MAX_EVENTS_CAP = 32;
    struct EventRecord {
        double snir         = 0.0;  // dB; meaningless unless snirValid
        double snirValid    = 0.0;
        double per          = 0.0;  // packet error rate, [0,1]
        double ber          = 0.0;  // bit error rate, [0,1]
        double size         = 0.0;  // bytes, as reported at the hook this
                                     // event was captured at (see .cc — MAC
                                     // header may or may not be popped yet
                                     // depending on which drop reason this is)
        double isDrop       = 0.0;
        double dropRetry    = 0.0;
        double dropQueue    = 0.0;
        double dropCorrupt  = 0.0;
        double dropNotForUs = 0.0;
        double fromDrone    = 0.0;
        double present      = 1.0;  // always 1.0 here — only real events are
                                     // ever pushed; Python pads with 0-rows
    };
    std::vector<std::vector<EventRecord>> droneEvents_;  // per drone, reset every RL step
    void recordEvent(int droneIdx, const EventRecord& rec);

    // CUT decision (OMNET.md S15): a real 802.11 MAC discards frames not
    // addressed to it before they reach the processor, so excluding them
    // models the actual sensor interface. Default true (cut). Set
    // OMNET_KEEP_NOTFORUS=1 to record the raw firehose instead — then also
    // raise MAX_EVENTS_CAP (see above). Read once in initialize().
    bool cutNotForUs_ = true;

    // ── Raw own-state sensors: channel occupancy + receive recency ─────────
    // channel_busy_fraction: a real CCA (clear channel assessment) measurement
    // from IRadio's reception state (IRadio::receptionStateChangedSignal,
    // subscribed per-drone directly on wlan[0].radio) — NOT derived from
    // packet counts. Busy time (RECEPTION_STATE_BUSY or _RECEIVING per
    // IRadio::ReceptionState) is accumulated between RL steps and reset every
    // step in buildObservationJson, exactly parallel to droneEvents_ above.
    std::vector<double>  droneBusyTime_;             // accumulated busy seconds this step
    std::vector<SimTime> droneLastStateChangeTime_;  // sim time of last known reception-state transition
    std::vector<int>     droneReceptionState_;       // last known IRadio::ReceptionState, as int

    // time_since_last_rx: seconds since this drone's radio last completed a
    // successful reception (the same radio-level packetSentToUpper hook
    // EventCounts::rx uses). Grows from episode start (SimTime::ZERO) if the
    // drone has never received anything yet — that IS the actual value, not
    // a sentinel.
    std::vector<SimTime> droneLastRxTime_;

    virtual void receiveSignal(cComponent* src, simsignal_t id,
                               intval_t value, cObject* details) override;

    // MAC address -> "sender is a drone" (true) / "sender is a user" (false).
    // Built once at INITSTAGE_TRANSPORT_LAYER (after INITSTAGE_LINK_LAYER,
    // where NetworkInterface auto-assigns real MAC addresses) from every
    // host's/user's wlan0 interface, so receiveSignal can classify an
    // event's sender without a live cross-module interface-table lookup.
    std::map<MacAddress, bool> macIsDrone_;

    // MAC address -> index, split by kind, built alongside macIsDrone_ above.
    // Lets receiveSignal resolve an arriving frame's transmitter to a
    // specific peer drone or user index (not just "is a drone"), needed to
    // attribute a measured RSSI reading to the right row below.
    std::map<MacAddress, int> macToDroneIdx_;
    std::map<MacAddress, int> macToUserIdx_;

    // ── Measured drone->peer / drone->user received power (RSSI) ───────────
    // Real measured received signal power (SignalPowerInd tag, attached by
    // the physical layer at the same point as SnirInd/ErrorRateInd — see
    // ReceiverBase::computeReceptionResult) from the LAST frame this drone
    // has actually received from each peer drone / user. Read at the same
    // "mac" packetSentToUpper hook already used for per-event SNIR
    // (receiveSignal(cObject*) below). This is NOT derived from positions —
    // if a drone has never received a frame from a given peer/user, the
    // corresponding *Valid_ entry stays 0.0 and the dBm entry stays 0.0;
    // that silence is the actual sensor state and callers must not
    // substitute a computed estimate for it.
    std::vector<std::vector<double>>  peerDroneRssiDbm_;    // [i][j] dBm; meaningless unless peerDroneRssiValid_[i][j]
    std::vector<std::vector<double>>  peerDroneRssiValid_;  // [i][j] 1.0/0.0
    std::vector<std::vector<SimTime>> peerDroneRssiTime_;   // [i][j] sim time of last measurement; SimTime::ZERO = never measured (grows from episode start, same convention as droneLastRxTime_ above)

    std::vector<std::vector<double>>  userRssiDbm_;         // [i][u] dBm; meaningless unless userRssiValid_[i][u]
    std::vector<std::vector<double>>  userRssiValid_;       // [i][u] 1.0/0.0
    std::vector<std::vector<SimTime>> userRssiTime_;        // [i][u] sim time of last measurement; SimTime::ZERO = never measured

    // Canvas figures
    std::vector<cLineFigure*> dronePairLines_;
    std::vector<cOvalFigure*> zoneFigures_;

    virtual void receiveSignal(cComponent* src, simsignal_t id,
                               cObject* obj, cObject* details) override;
};

} // namespace inet
#endif
