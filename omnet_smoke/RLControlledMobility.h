//
// RLControlledMobility.h — see .ned for module docs.
//
#ifndef __INET_RLCONTROLLEDMOBILITY_H
#define __INET_RLCONTROLLEDMOBILITY_H

#include "inet/mobility/base/MovingMobilityBase.h"

namespace inet {

class RLBridge;

class INET_API RLControlledMobility : public MovingMobilityBase
{
  public:
    RLControlledMobility();

  public:
    // Called by RLBridge at INITSTAGE_TRANSPORT_LAYER, after the user clusters
    // have been sampled, to place this drone relative to them. Overrides the
    // initialX/initialY read from the ini by MobilityBase. No-op for users.
    void setSpawnPosition(double x, double y);

  protected:
    virtual void initialize(int stage) override;
    virtual void move() override;

  private:
    int      droneId_    = -1;
    int      userId_     = -1;
    bool     isUserMode_ = false;
    double   maxSpeed_   = 0.0;
    RLBridge* bridge_    = nullptr;
};

} // namespace inet

#endif
