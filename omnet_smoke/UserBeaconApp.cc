//
// UserBeaconApp.cc  -- see UserBeaconApp.h for why this app exists.
//
#include "UserBeaconApp.h"

#include <cstdio>
#include <cstdint>
#include <vector>

#include "inet/common/ModuleAccess.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/chunk/BytesChunk.h"
#include "inet/networklayer/common/NetworkInterface.h"
#include "inet/networklayer/contract/IInterfaceTable.h"

namespace inet {

Define_Module(UserBeaconApp);

UserBeaconApp::~UserBeaconApp()
{
    cancelAndDelete(beaconTimer_);
}

void UserBeaconApp::initialize(int stage)
{
    cSimpleModule::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        userId_      = par("userId");
        if (userId_ < 0) {
            // Default path: the app is wired from the command line, where a
            // `parent.userId` NED expression is awkward to escape. Read the
            // containing UserHost's own id instead, so the app never has to
            // be told something the node already knows.
            cModule* host = getContainingNode(this);
            if (host && host->hasPar("userId"))
                userId_ = host->par("userId");
            else
                userId_ = host ? host->getIndex() : -1;
        }
        interval_    = par("beaconInterval");
        beaconPort_  = par("beaconPort");
        beaconBytes_ = par("beaconBytes");
        verbose_     = par("verbose");
        beaconGroup_ = Ipv4Address(par("beaconGroup").stringValue());

        beaconTimer_ = new cMessage("userBeaconTimer");
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        socket_.setOutputGate(gate("socketOut"));
        socket_.setCallback(this);
        socket_.bind(beaconPort_);

        // Pin the egress interface (MULTICAST_IF). User nodes run with
        // `ipv4.configurator.configureRoutingTable = false` and
        // `routingTable.netmaskRoutes = "lo*"`, so there is no wlan0 route to
        // resolve a multicast destination against; UserAssociationModule may
        // or may not have installed a default route by now, and the beacon
        // must not depend on that. With MULTICAST_IF set, Ipv4 takes the
        // "use the interface specified by MULTICAST_IF socket option" branch
        // in datagramLocalOut and never consults the routing table at all.
        cModule* host = getContainingNode(this);
        NetworkInterface* wlan = nullptr;
        if (host) {
            if (auto* iftMod = host->getSubmodule("interfaceTable")) {
                auto* ift = check_and_cast<IInterfaceTable*>(iftMod);
                wlan = ift->findInterfaceByName("wlan0");
                if (!wlan) {
                    // Fall back to the first non-loopback multicast-capable
                    // interface rather than silently sending nowhere.
                    for (int i = 0; i < ift->getNumInterfaces(); ++i) {
                        NetworkInterface* ie = ift->getInterface(i);
                        if (ie && !ie->isLoopback() && ie->isMulticast()) {
                            wlan = ie;
                            break;
                        }
                    }
                }
            }
        }
        if (!wlan)
            throw cRuntimeError("UserBeaconApp[%d]: no multicast-capable "
                                "interface found on the containing node", userId_);
        socket_.setMulticastOutputInterface(wlan->getInterfaceId());
        // No point looping our own presence beacon back up our own stack.
        socket_.setMulticastLoop(false);
        socket_.joinMulticastGroup(beaconGroup_);

        // Deterministic stagger, NOT a random one: drawing here would consume
        // from the module RNG stream and make a beacon-on run sample a
        // different episode layout than the same seed with the beacon off,
        // which would wreck any A/B against it.
        double frac = (userId_ >= 0) ? (double)(userId_ % 101) / 101.0 : 0.0;
        scheduleAt(simTime() + interval_ * frac + interval_, beaconTimer_);
    }
}

void UserBeaconApp::handleMessage(cMessage* msg)
{
    if (msg == beaconTimer_) {
        int numBytes = beaconBytes_ < 1 ? 1 : beaconBytes_;

        // Payload is pure presence: the drone attributes the frame by SOURCE
        // MAC at the radio hook (RLBridge's macToUserIdx_), so nothing in the
        // bytes is read by anyone. The first byte carries the low 8 bits of
        // the user id purely so a packet dump is readable.
        std::vector<uint8_t> payload(numBytes, 0x7E);
        payload[0] = (uint8_t)(userId_ & 0xFF);

        auto chunk = makeShared<BytesChunk>();
        chunk->setBytes(payload);

        auto* pkt = new Packet("userBeacon");
        pkt->insertAtBack(chunk);

        socket_.sendTo(pkt, beaconGroup_, beaconPort_);
        ++sentCount_;

        if (verbose_) {
            std::printf("USERBEACON:%.6f,%d,%d\n",
                        simTime().dbl(), userId_, numBytes);
            std::fflush(stdout);
        }

        EV_DETAIL << "UserBeaconApp[" << userId_ << "]: presence beacon sent to "
                  << beaconGroup_ << " (" << numBytes << " B)" << endl;

        scheduleAt(simTime() + interval_, beaconTimer_);
    }
    else {
        socket_.processMessage(msg);
    }
}

void UserBeaconApp::socketDataArrived(UdpSocket*, Packet* pkt)
{
    // Users do not act on each other's presence beacons; the drone side is
    // the only consumer, and it consumes them at the radio, not here.
    delete pkt;
}

void UserBeaconApp::finish()
{
    recordScalar("userBeaconsSent", sentCount_);
    socket_.close();
}

} // namespace inet
