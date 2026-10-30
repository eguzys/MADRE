//
// UserAssociationModule.h
//
// Phase A+: draws a solid cLineFigure on the network canvas to show each
// user's current drone association. Updated every evalInterval.
//
// Position-based BRP association (v1 simplification).
// Rather than sniffing UDP beacons (which require broadcast routing that
// conflicts with AODV's empty-routing-table setup), this module reads drone
// positions directly from RLBridge and computes RSSI via the same
// log-distance pathloss model used by OMNeT++/INET's LogNormalShadowing.
// This gives deterministic, noise-free RSSI for training stability.
//
#ifndef __INET_USERASSOCIATIONMODULE_H
#define __INET_USERASSOCIATIONMODULE_H

#include <vector>
#include <string>

#include "inet/common/INETDefs.h"
#include "inet/networklayer/contract/ipv4/Ipv4Address.h"

namespace omnetpp { class cLineFigure; } // forward-declare to avoid ccanvas.h in every TU

namespace inet {

class RLBridge;
class IInterfaceTable;
class IIpv4RoutingTable;
class NetworkInterface;
class MobilityBase;

/**
 * App-layer BRP (Biased Received Power) association module.
 *
 * Every evalInterval simulated seconds:
 *  1. Reads own position from parent host's StationaryMobility.
 *  2. Reads each drone position from RLBridge.
 *  3. Computes RSSI via log-distance pathloss (alpha=2.5, DRONE_TX=200mW).
 *  4. Computes BRP score: score[d] = RSSI[d] - beta * userCount[d].
 *  5. Switches default route if best candidate beats current by >= hysteresis.
 *  6. Notifies RLBridge on association change.
 */
class INET_API UserAssociationModule : public cSimpleModule
{
  public:
    virtual ~UserAssociationModule();

    int         getAssociatedDroneId()  const { return currentDrone_; }
    Ipv4Address getAssociatedDroneIp()  const;

  protected:
    virtual void initialize(int stage) override;
    virtual int  numInitStages() const override { return NUM_INIT_STAGES; }
    virtual void handleMessage(cMessage* msg) override;
    virtual void finish() override;

  private:
    int     userId_      = -1;
    double  alpha_       = 2.5;    // pathloss exponent (matches ini)
    double  droneTxMW_   = 200.0;  // drone TX power in mW  (downlink; ~525 m)
    double  userTxMW_    = 10.0;   // user  TX power in mW  (uplink;  ~158 m)
    // Association MUST be decided on the UPLINK, not the beacon. The beacon is a
    // 200 mW drone transmission (525 m); the user replies at 10 mW (158 m). Ranking
    // drones by beacon strength associates users with drones that physically cannot
    // hear them -> uplink silently fails, no packets, FES starves. This also aligns
    // the C++ model with max_flow_reward.py, which already computes c_up with
    // USER_TX_mW and c_down with DRONE_TX_mW separately (lines 189-190).
    double  beta_        = 3.0;    // dB per associated user
    double  hysteresis_  = 3.0;    // dB hysteresis threshold
    SimTime evalInterval_;
    int     currentDrone_ = -1;

    RLBridge*           bridge_       = nullptr;
    IInterfaceTable*    ifTable_      = nullptr;
    IIpv4RoutingTable*  routingTable_ = nullptr;
    MobilityBase*       myMobility_   = nullptr;

    std::vector<Ipv4Address> droneIpv4_;

    cMessage*    evalTimer_ = nullptr;
    cLineFigure* assocLine_ = nullptr;   // user→drone association line on network canvas

    double computeRssi_dBm(double dist_m) const;
    void   evaluateAssociation();
    void   setDefaultRoute(int droneId);
    void   resolvedroneIPs();
};

} // namespace inet
#endif
