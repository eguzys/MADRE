// UserBonnTrafficApp.cc — Bonn talk-group voice traffic (Aschenbruck IPCCC 2007 + LCN 2006)
//
// This module is a thin packet sender.  All conversation-process state and
// scheduling lives in RLBridge; this app only:
//   • binds a UDP socket and subscribes to its two multicast groups at init
//   • exposes sendMelpe(dest) which RLBridge calls when this user should speak
//   • logs every outgoing packet to stdout (TRAFFIC: prefix for tooling)
//
#include "inet/networklayer/common/HopLimitTag_m.h"
#include "UserBonnTrafficApp.h"
#include "RLBridge.h"

#include "inet/common/ModuleAccess.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/chunk/BytesChunk.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/contract/IInterfaceTable.h"

namespace inet {

Define_Module(UserBonnTrafficApp);

// ─── initialize ───────────────────────────────────────────────────────────────

void UserBonnTrafficApp::initialize(int stage)
{
    cSimpleModule::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        userId_               = par("userId");
        listenPort_           = par("listenPort");
        melpePayloadBytes_    = par("melpePayloadBytes");
        melpeIntervalMs_      = par("melpeInterval_ms");
        packetSizeMultiplier_ = par("packetSizeMultiplier");

        const char* bridgePath = par("bridgeModule").stringValue();
        cModule* m = getModuleByPath(bridgePath);
        if (!m)
            throw cRuntimeError("UserBonnTrafficApp[%d]: bridge not found at '%s'",
                                userId_, bridgePath);
        bridge_ = check_and_cast<RLBridge*>(m);
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        socket_.setOutputGate(gate("socketOut"));
        socket_.setCallback(this);
        socket_.bind(listenPort_);

        // Subscribe to local-area multicast group and the global group.
        int areaId = bridge_->getUserArea(userId_);
        std::string localMcastStr = "239.1.1." + std::to_string(areaId + 1);
        socket_.joinMulticastGroup(Ipv4Address(localMcastStr.c_str()));
        socket_.joinMulticastGroup(Ipv4Address("239.1.1.10"));

        // Pass our parameters to RLBridge so it can seed the conversation process.
        TrafficParams tp;
        tp.headInitiatorProb    = par("headInitiatorProb");
        tp.meshProb             = par("meshProb");
        tp.transHeadToHead      = par("transHeadToHead");
        tp.transMemberToMember  = par("transMemberToMember");
        tp.callContinuationProb = par("callContinuationProb");
        tp.callCHT_meanlog      = par("callCHT_meanlog");
        tp.callCHT_sdlog        = par("callCHT_sdlog");
        tp.callIdle_shape       = par("callIdle_shape");
        tp.callIdle_rate        = par("callIdle_rate");
        tp.convIdle_meanlog     = par("convIdle_meanlog");
        tp.convIdle_sdlog       = par("convIdle_sdlog");
        tp.convIdle_shift       = par("convIdle_shift");
        tp.melpeIntervalMs      = par("melpeInterval_ms");
        tp.paramsLoaded         = true;
        bridge_->registerTrafficApp(userId_, this, tp);

        EV_INFO << "UserBonnTrafficApp[" << userId_ << "]: area=" << areaId
                << " joined " << localMcastStr << " + 239.1.1.10" << endl;
    }
}

// ─── sendMelpe ────────────────────────────────────────────────────────────────

void UserBonnTrafficApp::sendMelpe(Ipv4Address dest)
{
    Enter_Method_Silent();
    int numBytes = melpePayloadBytes_ * packetSizeMultiplier_;
    if (numBytes < 1) numBytes = 1;

    auto chunk = makeShared<BytesChunk>();
    chunk->setBytes(std::vector<uint8_t>(numBytes, 0xAB));
    auto* pkt = new Packet("melpePkt");
    pkt->insertAtBack(chunk);
    socket_.sendTo(pkt, dest, listenPort_);

    std::printf("TRAFFIC:%.6f,%d,%s,%d\n",
                simTime().dbl(), userId_, dest.str().c_str(), numBytes);
    std::fflush(stdout);
}

// ─── handleMessage / socketDataArrived / finish ───────────────────────────────

void UserBonnTrafficApp::handleMessage(cMessage* msg)
{
    if (!msg->isSelfMessage()) {
        socket_.processMessage(msg);
        return;
    }
    EV_WARN << "UserBonnTrafficApp[" << userId_ << "]: unexpected self-msg" << endl;
    delete msg;
}

void UserBonnTrafficApp::socketDataArrived(UdpSocket*, Packet* pkt)
{
    // RELAY: how many L3 hops did this packet take? A packet that went
    // user -> drone -> user has hopLimit decremented by the drones it traversed.
    // INET default TTL is 32; 32 = direct (no relay), 31 = one drone, 30 = two.
    int ttl = -1;
    auto ind = pkt->findTag<HopLimitInd>();
    if (ind) ttl = ind->getHopLimit();
    std::printf("DELIVERED:%.6f,%d,%ld,ttl=%d\n",
                simTime().dbl(), userId_, (long)pkt->getByteLength(), ttl);
    std::fflush(stdout);
    delete pkt;
}

void UserBonnTrafficApp::finish()
{
    socket_.close();
}

} // namespace inet
