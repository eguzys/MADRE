//
// UserAssociationModule.cc
//
// Position-based BRP association. Reads drone positions from RLBridge
// and computes RSSI via log-distance model matching OMNeT++ LogNormalShadowing.
// Route discovery is handled by AODV; this module only tracks association
// telemetry (droneUserCounts_) exposed in the obs JSON.
//
#include <cstdio>
#include "UserAssociationModule.h"
#include "RLBridge.h"

#include "inet/common/ModuleAccess.h"
#include "inet/mobility/base/MobilityBase.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/contract/IInterfaceTable.h"
#include "inet/networklayer/ipv4/IIpv4RoutingTable.h"
#include "inet/networklayer/ipv4/Ipv4Route.h"

#include <omnetpp/ccanvas.h>   // cLineFigure

#include <cmath>
#include <limits>

namespace inet {

Define_Module(UserAssociationModule);

static constexpr double FREQ_HZ         = 2.4e9;
static constexpr double LIGHT_SPEED_MPS = 299792458.0;

UserAssociationModule::~UserAssociationModule()
{
    cancelAndDelete(evalTimer_);
}

void UserAssociationModule::initialize(int stage)
{
    cSimpleModule::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        userId_       = par("userId");
        beta_         = par("beta");
        hysteresis_   = par("hysteresis");
        evalInterval_ = par("evalInterval");

        const char* bridgePath = par("bridgeModule").stringValue();
        cModule* m = getModuleByPath(bridgePath);
        if (!m)
            throw cRuntimeError("UserAssociationModule[%d]: bridge not found at '%s'",
                                userId_, bridgePath);
        bridge_ = check_and_cast<RLBridge*>(m);

        evalTimer_ = new cMessage("evalTimer");
    }
    else if (stage == INITSTAGE_NETWORK_LAYER) {
        cModule* host = getContainingNode(this);
        ifTable_ = check_and_cast<IInterfaceTable*>(
            host->getSubmodule("interfaceTable"));
        routingTable_ = check_and_cast<IIpv4RoutingTable*>(
            host->getModuleByPath(".ipv4.routingTable"));
        myMobility_ = check_and_cast<MobilityBase*>(
            host->getSubmodule("mobility"));
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        resolvedroneIPs();

        // ── Strip auto-installed interface routes ─────────────────────────
        // INET's IPv4 stack auto-installs an IFACENETMASK /16 route on wlan0
        // when the configurator assigns an address. Without removing it, peer
        // users would match the /16 (longest prefix wins over the default
        // route), attempt direct ARP, and bypass the drone backbone. Strip it
        // so the only non-loopback route is the default route installed by
        // setDefaultRoute → forces all outbound traffic via the drone.
        {
            NetworkInterface* stripWlan = ifTable_->findInterfaceByName("wlan0");
            if (stripWlan) {
                for (int i = 0; i < routingTable_->getNumRoutes(); ) {
                    Ipv4Route* r = routingTable_->getRoute(i);
                    if (r->getSourceType() == IRoute::IFACENETMASK
                        && r->getInterface() == stripWlan) {
                        routingTable_->deleteRoute(r);
                    } else { ++i; }
                }
            }
        }
        std::printf("ASSOC_INIT: user=%d numDrones=%d evalInterval=%.3f\n",
                    userId_, (int)droneIpv4_.size(), evalInterval_.dbl());
        std::fflush(stdout);
        scheduleAt(simTime() + evalInterval_, evalTimer_);

        // Create solid user→drone association line on network canvas (Phase A+).
        cCanvas* canvas = getSystemModule()->getCanvas();
        std::string lineName = "userAssocLine_" + std::to_string(userId_);
        assocLine_ = new cLineFigure(lineName.c_str());
        assocLine_->setLineColor(cFigure::Color("#3366FF"));
        assocLine_->setLineWidth(1.5);
        assocLine_->setZIndex(-0.2);
        assocLine_->setVisible(false);
        canvas->addFigure(assocLine_);
    }
}

void UserAssociationModule::resolvedroneIPs()
{
    cModule* net = getParentModule()->getParentModule();
    int numDrones = (int)net->par("numDrones").intValue();
    droneIpv4_.resize(numDrones, Ipv4Address::UNSPECIFIED_ADDRESS);

    for (int i = 0; i < numDrones; i++) {
        cModule* droneHost = net->getSubmodule("host", i);
        if (!droneHost) continue;
        auto* ift = L3AddressResolver().findInterfaceTableOf(droneHost);
        if (!ift) continue;
        L3Address a = L3AddressResolver().getAddressFrom(ift);
        if (!a.isUnspecified())
            droneIpv4_[i] = a.toIpv4();
    }
}

double UserAssociationModule::computeRssi_dBm(double dist_m) const
{
    if (dist_m < 1.0) dist_m = 1.0;
    double lam    = LIGHT_SPEED_MPS / FREQ_HZ;
    double fspl1m = 20.0 * std::log10(4.0 * M_PI / lam);
    double pl_dB  = fspl1m + 10.0 * alpha_ * std::log10(dist_m);
    // UPLINK: the binding constraint is whether the DRONE can hear the USER.
    double tx_dBm = 10.0 * std::log10(userTxMW_);
    return tx_dBm - pl_dB;
}

void UserAssociationModule::handleMessage(cMessage* msg)
{
    if (msg == evalTimer_) {
        evaluateAssociation();
        scheduleAt(simTime() + evalInterval_, evalTimer_);
    }
    else {
        EV_WARN << "UserAssociationModule[" << userId_
                << "]: unexpected message " << msg->getName() << endl;
        delete msg;
    }
}

void UserAssociationModule::evaluateAssociation()
{
    int numDrones = (int)droneIpv4_.size();
    if (numDrones == 0) return;

    Coord myPos = myMobility_->getCurrentPosition();

    int    bestDrone = -1;
    double bestScore = -std::numeric_limits<double>::infinity();

    for (int d = 0; d < numDrones; d++) {
        double droneX = 0.0, droneY = 0.0;
        bridge_->getDronePosition(d, droneX, droneY);
        double dist  = std::sqrt((myPos.x - droneX) * (myPos.x - droneX)
                               + (myPos.y - droneY) * (myPos.y - droneY));
        double rssi  = computeRssi_dBm(dist);
        double score = rssi - beta_ * static_cast<double>(bridge_->getUserCount(d));
        if (score > bestScore) { bestScore = score; bestDrone = d; }
    }

    if (bestDrone == -1) return;

    if (currentDrone_ == -1) {
        EV_INFO << "UserAssociationModule[" << userId_
                << "]: initial association → drone " << bestDrone
                << " (score=" << bestScore << " dBm)" << endl;
        setDefaultRoute(bestDrone);
        currentDrone_ = bestDrone;
    }
    else {
        double curDroneX = 0.0, curDroneY = 0.0;
        bridge_->getDronePosition(currentDrone_, curDroneX, curDroneY);
        double curDist  = std::sqrt((myPos.x - curDroneX) * (myPos.x - curDroneX)
                                  + (myPos.y - curDroneY) * (myPos.y - curDroneY));
        double curScore = computeRssi_dBm(curDist)
                        - beta_ * (double)bridge_->getUserCount(currentDrone_);

        if (bestScore > curScore + hysteresis_) {
            EV_INFO << "UserAssociationModule[" << userId_
                    << "]: handoff " << currentDrone_ << " → " << bestDrone
                    << " (gain=" << (bestScore - curScore) << " dB)" << endl;
            setDefaultRoute(bestDrone);
            currentDrone_ = bestDrone;
        }
    }

    // Refresh association line endpoint (drone may have moved since last eval).
    if (currentDrone_ >= 0 && assocLine_) {
        Coord myPos = myMobility_->getCurrentPosition();
        double droneX = 0.0, droneY = 0.0;
        bridge_->getDronePosition(currentDrone_, droneX, droneY);
        assocLine_->setStart(cFigure::Point(myPos.x, myPos.y));
        assocLine_->setEnd(cFigure::Point(droneX, droneY));
        assocLine_->setVisible(true);
    }
}

void UserAssociationModule::setDefaultRoute(int droneId)
{
    if (droneId < 0 || droneId >= (int)droneIpv4_.size()) return;

    Ipv4Address gw = droneIpv4_[droneId];
    std::printf("ASSOC_DIAG:%.3f user=%d -> drone=%d gw=%s\n",
                simTime().dbl(), userId_, droneId, gw.str().c_str());
    std::fflush(stdout);
    if (gw.isUnspecified()) {
        EV_WARN << "UserAssociationModule[" << userId_
                << "]: drone " << droneId << " has no IPv4 yet" << endl;
        return;
    }

    NetworkInterface* wlan0 = ifTable_->findInterfaceByName("wlan0");
    if (!wlan0) {
        EV_WARN << "UserAssociationModule[" << userId_ << "]: wlan0 not found" << endl;
        return;
    }

    // Remove old MANUAL default route.
    for (int i = 0; i < routingTable_->getNumRoutes(); ) {
        Ipv4Route* r = routingTable_->getRoute(i);
        if (r->getDestination().isUnspecified()
                && r->getNetmask().isUnspecified()
                && r->getSourceType() == IRoute::MANUAL) {
            routingTable_->deleteRoute(r);
        } else { ++i; }
    }

    // Install new MANUAL default route toward the associated drone.
    auto* userRoute = new Ipv4Route();
    userRoute->setDestination(Ipv4Address::UNSPECIFIED_ADDRESS);
    userRoute->setNetmask(Ipv4Address::UNSPECIFIED_ADDRESS);
    userRoute->setGateway(gw);
    userRoute->setInterface(wlan0);
    userRoute->setSourceType(IRoute::MANUAL);
    userRoute->setMetric(1);
    routingTable_->addRoute(userRoute);

    // RT_DUMP: table AFTER the default route is installed.
    if (userId_ == 6) {
        static bool dumped = false;
        if (!dumped) {
            dumped = true;
            std::printf("RT_DUMP: user6 POST-INSTALL forwarding=%d nRoutes=%d\n",
                        (int)routingTable_->isForwardingEnabled(),
                        routingTable_->getNumRoutes());
            for (int i = 0; i < routingTable_->getNumRoutes(); ++i) {
                Ipv4Route* r = routingTable_->getRoute(i);
                std::printf("RT_DUMP:   dst=%-15s mask=%-15s gw=%-12s if=%s src=%d\n",
                            r->getDestination().str().c_str(),
                            r->getNetmask().str().c_str(),
                            r->getGateway().str().c_str(),
                            r->getInterface() ? r->getInterface()->getInterfaceName() : "?",
                            (int)r->getSourceType());
            }
            // and where does 10.0.0.8 actually resolve to?
            Ipv4Route* hit = const_cast<Ipv4Route*>(
                routingTable_->findBestMatchingRoute(Ipv4Address("10.0.0.8")));
            if (hit)
                std::printf("RT_DUMP: lookup 10.0.0.8 -> gw=%s if=%s\n",
                            hit->getGateway().str().c_str(),
                            hit->getInterface() ? hit->getInterface()->getInterfaceName() : "?");
            else
                std::printf("RT_DUMP: lookup 10.0.0.8 -> NO MATCHING ROUTE\n");
            std::fflush(stdout);
        }
    }

    EV_INFO << "UserAssociationModule[" << userId_
            << "]: default route → " << gw << " (drone " << droneId << ")" << endl;

    // Notify RLBridge to update the drone-side /32 entries.
    Ipv4Address myIp = wlan0->getIpv4Address();
    if (!myIp.isUnspecified())
        bridge_->notifyAllDronesOfAssociation(userId_, myIp, droneId, currentDrone_);
}

Ipv4Address UserAssociationModule::getAssociatedDroneIp() const
{
    if (currentDrone_ < 0 || currentDrone_ >= (int)droneIpv4_.size())
        return Ipv4Address::UNSPECIFIED_ADDRESS;
    return droneIpv4_[currentDrone_];
}

void UserAssociationModule::finish() {}

} // namespace inet
