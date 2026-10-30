//
// RLControlledMobility.cc
//
#include "RLControlledMobility.h"
#include "RLBridge.h"
#include "inet/common/INETMath.h"

#include <cmath>

namespace inet {

Define_Module(RLControlledMobility);

RLControlledMobility::RLControlledMobility() = default;

void RLControlledMobility::initialize(int stage)
{
    MovingMobilityBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        droneId_  = par("droneId");
        maxSpeed_ = par("maxSpeed").doubleValue();

        // Resolve and cast the bridge module from the NED path
        const char* bridgePath = par("bridgeModule").stringValue();
        cModule* m = getModuleByPath(bridgePath);
        if (!m)
            throw cRuntimeError("RLControlledMobility[%d]: bridge module "
                                "not found at path '%s'", droneId_, bridgePath);
        bridge_ = check_and_cast<RLBridge*>(m);
        bridge_->registerDrone(droneId_, this);

        // We have no velocity until Python sends the first action.
        lastVelocity = Coord::ZERO;
    }
}

void RLControlledMobility::move()
{
    // Read the latest action from the bridge.
    double dx = 0.0, dy = 0.0;
    bridge_->getAction(droneId_, dx, dy);

    if (std::isnan(dx) || std::isnan(dy)) {
        // No action received yet (before first sync). Hold position.
        lastVelocity = Coord::ZERO;
    }
    else {
        // (dx, dy) is interpreted as a velocity in m/s applied across the
        // next sync interval. Clip to maxSpeed_.
        double mag = std::sqrt(dx*dx + dy*dy);
        if (mag > maxSpeed_ && mag > 0.0) {
            dx *= maxSpeed_ / mag;
            dy *= maxSpeed_ / mag;
        }
        lastVelocity = Coord(dx, dy, 0.0);
    }

    // Standard integrate-velocity step, identical to LinearMobility.
    double elapsedTime = (simTime() - lastUpdate).dbl();
    lastPosition += lastVelocity * elapsedTime;

    Coord dummy;
    handleIfOutside(REFLECT, dummy, lastVelocity);
}

} // namespace inet
