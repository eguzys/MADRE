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

  protected:
    virtual void initialize(int stage) override;
    virtual void move() override;

  private:
    int droneId_ = -1;
    double maxSpeed_ = 0.0;
    RLBridge* bridge_ = nullptr;
};

} // namespace inet

#endif
