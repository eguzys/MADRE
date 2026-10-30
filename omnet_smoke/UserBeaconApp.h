//
// UserBeaconApp.h
//
// PRESENCE BEACON for ground users.
//
// Why this exists: on the OM2G track a user only ever puts a frame on the
// medium while it is the CURRENT SPEAKER of a conversation
// (UserBonnTrafficApp::sendMelpe, driven by RLBridge's call engine). Only
// users drawn as "senders" get a conversation at all (RLBridge's
// buildPadreSenders draws each user independently with senderProb), and a
// sender's PARTNER never speaks either ("PADRE traffic: sender is fixed, no
// speaker alternation"). A non-calling user is therefore RADIO-SILENT, and a
// drone -- whose only sensing channel is the received frame (RLBridge's
// packetSentToUpper hook on the radio, macToUserIdx_) -- cannot know it is
// there at all.
//
// This app gives every user, calling or not, a small periodic multicast frame
// so that presence is observable. Nothing on the drone side changes: the
// existing radio hook already sets userRssiDbm_/userRssiValid_/userRssiTime_
// for ANY completed reception whose source MAC maps to a user.
//
// Modelled on DroneBeaconApp; multicast rather than broadcast for the same
// reason (a broadcast needs a routing table these user nodes deliberately do
// not have -- see om2g.ini's configureRoutingTable=false / netmaskRoutes="lo*").
// The egress interface is pinned explicitly with the MULTICAST_IF socket
// option so the send never depends on what UserAssociationModule happens to
// have installed in the routing table at that instant.
//
// OFF BY DEFAULT and only ever instantiated when $OM2G_USER_BEACON=1 makes
// om2g_config.om2g_extra_args() emit the app[2] assignments.
//
#ifndef __INET_USERBEACONAPP_H
#define __INET_USERBEACONAPP_H

#include "inet/common/INETDefs.h"
#include "inet/networklayer/contract/ipv4/Ipv4Address.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"

namespace inet {

class INET_API UserBeaconApp : public cSimpleModule, public UdpSocket::ICallback
{
  public:
    virtual ~UserBeaconApp();

  protected:
    virtual void initialize(int stage) override;
    virtual int  numInitStages() const override { return NUM_INIT_STAGES; }
    virtual void handleMessage(cMessage* msg) override;
    virtual void finish() override;

    virtual void socketDataArrived(UdpSocket*, Packet* pkt) override;
    virtual void socketErrorArrived(UdpSocket*, Indication*) override {}
    virtual void socketClosed(UdpSocket*) override {}

  private:
    int         userId_      = -1;
    SimTime     interval_;
    int         beaconPort_  = 5003;     // distinct from the drone beacon's 5002
    int         beaconBytes_ = 1;
    bool        verbose_     = false;
    Ipv4Address beaconGroup_;            // distinct from the drone's 239.1.2.1

    long        sentCount_   = 0;
    cMessage*   beaconTimer_ = nullptr;
    UdpSocket   socket_;
};

} // namespace inet
#endif
