//
// JammerGradientMobility — RF-power gradient-seeking mobility for the jammer.
//
// Phases (step = floor(simTime / syncInterval)):
//   [0, startMoving)        hold spawn position (random in arena)
//   [startMoving, activate) gradient ascent toward highest received power
//   [activate, ∞)           freeze; activates JammerNoiseSource emission
//
package rlsmoke;

import inet.mobility.base.MovingMobilityBase;

simple JammerGradientMobility extends MovingMobilityBase
{
    parameters:
        @class(inet::JammerGradientMobility);
        @display("i=block/control");

        // Path to the RLBridge module — single step-counter source of truth (Task 3).
        // Same convention as RLControlledMobility: two levels up from jammer.mobility.
        string bridgeModule = default("^.^.rlBridge");

        // Absolute path to the radio medium module (resolves from system root).
        string radioMediumModule = default("radioMedium");

        // Step at which gradient ascent begins (hold before this).
        int startMoving = default(100);

        // Step at which the jammer freezes and activates noise emission.
        int activate = default(500);

        // Number of compass-direction probe points for gradient sampling.
        int numSamples = default(8);

        // Distance of each probe point from current position (m).
        double stepSize @unit(m) = default(20m);

        // RL sync interval — used to derive step from simTime (must match
        // *.rlBridge.syncInterval in omnetpp.ini).
        double syncInterval @unit(s) = default(1s);

        // Update once per RL tick (matches syncInterval).
        updateInterval @unit(s) = default(1s);
}
