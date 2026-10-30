// UserBonnTrafficApp.h — Bonn talk-group voice traffic (Aschenbruck IPCCC 2007 + LCN 2006)
//
// Sends fixed-size MELPE voice packets (21 bytes × multiplier) to talk-group
// multicast addresses when commanded by RLBridge's conversation process.
// Subscribes to two multicast groups at init: local area + global (239.1.1.10).
#ifndef __INET_USERTRAFFIC_H
#define __INET_USERTRAFFIC_H

#include "inet/common/INETDefs.h"
#include "inet/networklayer/contract/ipv4/Ipv4Address.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"

namespace inet {

class RLBridge;

class INET_API UserBonnTrafficApp : public cSimpleModule, public UdpSocket::ICallback
{
  public:
    virtual ~UserBonnTrafficApp() = default;

    // Called by RLBridge conversation process to emit one MELPE frame.
    void sendMelpe(Ipv4Address dest);

  protected:
    virtual void initialize(int stage) override;
    virtual int  numInitStages() const override { return NUM_INIT_STAGES; }
    virtual void handleMessage(cMessage* msg) override;
    virtual void finish() override;

    virtual void socketDataArrived(UdpSocket*, Packet* pkt) override;
    virtual void socketErrorArrived(UdpSocket*, Indication*) override {}
    virtual void socketClosed(UdpSocket*) override {}

  private:
    int       userId_              = -1;
    int       listenPort_          = 5001;
    int       melpePayloadBytes_   = 21;
    double    melpeIntervalMs_     = 67.5;
    int       packetSizeMultiplier_= 1;
    RLBridge* bridge_              = nullptr;

    UdpSocket socket_;
};

} // namespace inet
#endif
