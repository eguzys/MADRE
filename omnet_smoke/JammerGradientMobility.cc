// JammerGradientMobility.cc
#include "JammerGradientMobility.h"
#include "JammerNoiseSource.h"
#include "RLBridge.h"

#include "inet/physicallayer/wireless/common/contract/packetlevel/ICommunicationCache.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IRadio.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/ITransmitter.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IAntenna.h"
#include "inet/mobility/contract/IMobility.h"

#include <cmath>

namespace inet {

using namespace physicallayer;

Define_Module(JammerGradientMobility);

void JammerGradientMobility::initialize(int stage)
{
    MovingMobilityBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        startMoving_  = (int)par("startMoving").intValue();
        activate_     = (int)par("activate").intValue();
        numSamples_   = (int)par("numSamples").intValue();
        stepSize_     = par("stepSize").doubleValue();

        // Parent module IS the JammerNoiseSource (jammer.mobility → jammer).
        noiseSource_ = check_and_cast<JammerNoiseSource*>(getParentModule());

        // Single source of truth for step count: read from RLBridge (Task 3).
        // bridgeModule path mirrors RLControlledMobility convention ("^.^.rlBridge").
        const char* brPath = par("bridgeModule").stringValue();
        cModule* brMod = getModuleByPath(brPath);
        if (!brMod)
            throw cRuntimeError("JammerGradientMobility: rlBridge not found at '%s'", brPath);
        bridge_ = check_and_cast<RLBridge*>(brMod);

        // Radio medium: absolute path "radioMedium" resolves to SmokeNet.radioMedium.
        const char* rmPath = par("radioMediumModule").stringValue();
        cModule* rm = getModuleByPath(rmPath);
        if (!rm)
            throw cRuntimeError("JammerGradientMobility: radioMedium not found at '%s'", rmPath);
        radioMedium_ = check_and_cast<IRadioMedium*>(rm);
    }
}

void JammerGradientMobility::move()
{
    // Single source of truth: read step counter from RLBridge (Task 3).
    int step = bridge_->getStepCounter();

    lastVelocity = Coord::ZERO;

    if (step < startMoving_) {
        // Phase 1: hold spawn position.
        return;
    }

    if (step >= activate_) {
        // Phase 3: freeze and activate jammer noise emission (once).
        if (!activated_) {
            activated_ = true;
            noiseSource_->activateJammer();
            EV_INFO << "JammerGradientMobility: ACTIVATED at step=" << step
                    << " pos=(" << lastPosition.x << "," << lastPosition.y << ")" << endl;
        }
        return;
    }

    // Phase 2: gradient ascent — one move per RL tick.
    if (step == lastStep_) return;
    lastStep_ = step;

    // Sample power at numSamples_ equally-spaced compass directions.
    static const double TWO_PI = 2.0 * M_PI;
    double bestPower = -1.0;
    int    bestDir   = -1;

    for (int d = 0; d < numSamples_; ++d) {
        double angle = TWO_PI * d / numSamples_;
        Coord probe = lastPosition + Coord(stepSize_ * std::cos(angle),
                                          stepSize_ * std::sin(angle),
                                          0.0);
        // Clamp probe to constraint area.
        probe.x = std::max(constraintAreaMin.x, std::min(constraintAreaMax.x, probe.x));
        probe.y = std::max(constraintAreaMin.y, std::min(constraintAreaMax.y, probe.y));

        double power = computeTotalPowerAt(probe);
        if (power > bestPower) { bestPower = power; bestDir = d; }
    }

    // Move toward the highest-power direction.
    if (bestDir >= 0 && bestPower > 0.0) {
        double angle = TWO_PI * bestDir / numSamples_;
        lastPosition.x += stepSize_ * std::cos(angle);
        lastPosition.y += stepSize_ * std::sin(angle);
        // Clamp to arena.
        lastPosition.x = std::max(constraintAreaMin.x, std::min(constraintAreaMax.x, lastPosition.x));
        lastPosition.y = std::max(constraintAreaMin.y, std::min(constraintAreaMax.y, lastPosition.y));
    }

    EV_INFO << "JammerGradientMobility: step=" << step
            << " pos=(" << lastPosition.x << "," << lastPosition.y
            << ") bestPow=" << bestPower << " W" << endl;
}

double JammerGradientMobility::computeTotalPowerAt(const Coord& probe)
{
    // Deterministic log-distance path loss at 2.4 GHz, alpha=2.5, d0=1 m:
    //   loss(d) = λ²/(16π²·d^alpha)   where λ = c/f = 3e8/2.4e9 ≈ 0.125 m
    static const double kLambda  = 3e8 / 2.4e9;
    static const double kLossD0  = (kLambda * kLambda) / (16.0 * M_PI * M_PI);
    static const double kAlpha   = 2.5;

    double total = 0.0;
    const ICommunicationCache* cache = radioMedium_->getCommunicationCache();

    cache->mapRadios([&](const IRadio* radio) {
        const ITransmitter* tx = radio->getTransmitter();
        if (!tx) return;

        W maxPow = tx->getMaxPower();
        if (std::isnan(maxPow.get())) return;  // skip jammer (NoiseTransmitter returns NaN)

        const IAntenna* ant = radio->getAntenna();
        if (!ant || !ant->getMobility()) return;

        Coord txPos = ant->getMobility()->getCurrentPosition();
        double d = probe.distance(txPos);
        if (d < 1.0) d = 1.0;

        double loss = kLossD0 / std::pow(d, kAlpha);
        total += maxPow.get() * loss;
    });

    return total;
}

} // namespace inet
