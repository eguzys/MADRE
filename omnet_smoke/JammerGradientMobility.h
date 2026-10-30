// JammerGradientMobility.h
//
// Gradient-seeking mobility for the RF jammer.
//
// Phases (keyed on RL step = floor(simTime / syncInterval)):
//   [0, startMoving)   — hold spawn position
//   [startMoving, activate) — gradient ascent: sample received power at
//                          numSamples compass-direction probe points, move
//                          toward the highest. One move per RL tick.
//   [activate, ∞)      — freeze; activates JammerNoiseSource emission.
//
// Power at a probe point is computed analytically from all registered radios
// using deterministic log-distance path loss (alpha=2.5, 2.4 GHz, d0=1m),
// which avoids random shadowing draws in the gradient loop.
//
#ifndef JAMMERGRADIENTMOBILITY_H
#define JAMMERGRADIENTMOBILITY_H

#include "inet/mobility/base/MovingMobilityBase.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IRadioMedium.h"

namespace inet {

class JammerNoiseSource;
class RLBridge;

class INET_API JammerGradientMobility : public MovingMobilityBase
{
  public:
    JammerGradientMobility() = default;

  protected:
    virtual void initialize(int stage) override;
    virtual void move() override;

  private:
    int     startMoving_ = 100;
    int     activate_    = 500;
    int     numSamples_  = 8;
    double  stepSize_    = 20.0;   // meters

    int     lastStep_  = -1;
    bool    activated_ = false;

    JammerNoiseSource*              noiseSource_ = nullptr;
    RLBridge*                       bridge_      = nullptr;  // single step-counter source
    const physicallayer::IRadioMedium* radioMedium_ = nullptr;

    // Returns summed deterministic received power (W) from all registered
    // radios at the given probe point. Uses log-distance path loss with
    // alpha=2.5, 2.4 GHz, no shadowing.
    double computeTotalPowerAt(const Coord& probe);
};

} // namespace inet

#endif
