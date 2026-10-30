//
// DroneBeaconApp.h
//
// Periodic hello beacon: every beaconInterval, sends a small UDP packet to a
// drone-only multicast group so peers can hear it (real measured peer RSSI)
// and AODV can discover routes through the drone backbone.
//
#ifndef __INET_DRONEBEACONAPP_H
#define __INET_DRONEBEACONAPP_H

#include "inet/common/INETDefs.h"
#include "inet/networklayer/contract/ipv4/Ipv4Address.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"

namespace inet {

class RLBridge;

class INET_API DroneBeaconApp : public cSimpleModule, public UdpSocket::ICallback
{
  public:
    virtual ~DroneBeaconApp();

  protected:
    virtual void initialize(int stage) override;
    virtual int  numInitStages() const override { return NUM_INIT_STAGES; }
    virtual void handleMessage(cMessage* msg) override;
    virtual void finish() override;

    virtual void socketDataArrived(UdpSocket*, Packet* pkt) override;
    virtual void socketErrorArrived(UdpSocket*, Indication*) override {}
    virtual void socketClosed(UdpSocket*) override {}

  private:
    int         droneId_     = -1;
    SimTime     interval_;
    int         beaconPort_  = 5002;
    int         beaconBytes_ = 32;
    bool        dynamicBeaconSize_  = false;
    int         beaconFixedBytes_   = 56;
    int         beaconBytesPerUser_ = 52;
    Ipv4Address beaconGroup_;

    // Opt-in "cheap" send-gating heuristic: skip this beacon entirely
    // (no packet, no airtime charge) if nothing worth reporting has changed
    // since the last one actually sent. Default off reproduces every
    // existing run byte-identical. See DroneBeaconApp.cc.
    bool   sendOnDeltaEnabled_       = false;
    double sendOnDeltaPosThresholdM_ = 10.0;
    bool   learnedGateEnabled_       = false;
    bool   everSent_          = false;
    double lastSentX_         = 0.0;
    double lastSentY_         = 0.0;
    int    lastSentSensedCount_ = -1;
    RLBridge*   bridge_      = nullptr;
    cMessage*   beaconTimer_ = nullptr;

    UdpSocket   socket_;
};

} // namespace inet
#endif
