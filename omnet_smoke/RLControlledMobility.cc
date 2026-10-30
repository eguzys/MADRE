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
        droneId_ = par("droneId");
        userId_  = par("userId");
        maxSpeed_ = par("maxSpeed").doubleValue();

        const char* bridgePath = par("bridgeModule").stringValue();
        cModule* m = getModuleByPath(bridgePath);
        if (!m)
            throw cRuntimeError("RLControlledMobility: bridge module "
                                "not found at path '%s'", bridgePath);
        bridge_ = check_and_cast<RLBridge*>(m);

        if (userId_ >= 0) {
            isUserMode_ = true;
            // No registration needed; bridge pre-samples positions at
            // INITSTAGE_TRANSPORT_LAYER and move() reads them directly.
        } else if (droneId_ >= 0) {
            bridge_->registerDrone(droneId_, this);
        } else {
            throw cRuntimeError("RLControlledMobility: both droneId and userId are -1");
        }

        lastVelocity = Coord::ZERO;
    }
}

void RLControlledMobility::setSpawnPosition(double x, double y)
{
    if (isUserMode_) return;                 // users are placed via getSampledUserPosition
    lastPosition = Coord(x, y, 0.0);
    lastVelocity = Coord::ZERO;
    emitMobilityStateChangedSignal();        // radios/visualizer must see the new position
}

void RLControlledMobility::move()
{
    if (isUserMode_) {
        // Hold sampled position for the entire episode (stationary v1).
        lastPosition = bridge_->getSampledUserPosition(userId_);
        lastVelocity = Coord::ZERO;
        return;
    }

    // Drone mode: read velocity action from bridge.
    double dx = 0.0, dy = 0.0;
    bridge_->getAction(droneId_, dx, dy);

    if (std::isnan(dx) || std::isnan(dy)) {
        lastVelocity = Coord::ZERO;
    }
    else {
        double mag = std::sqrt(dx*dx + dy*dy);
        if (mag > maxSpeed_ && mag > 0.0) {
            dx *= maxSpeed_ / mag;
            dy *= maxSpeed_ / mag;
        }
        lastVelocity = Coord(dx, dy, 0.0);
    }

    double elapsedTime = (simTime() - lastUpdate).dbl();
    lastPosition += lastVelocity * elapsedTime;

    Coord dummy;
    handleIfOutside(REFLECT, dummy, lastVelocity);
}

} // namespace inet
