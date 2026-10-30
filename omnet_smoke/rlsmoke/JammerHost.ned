//
// JammerHost — gradient-seeking RF jammer node.
//
// Extends INET's NoiseSource with:
//   • JammerGradientMobility: moves toward highest RF power during
//     steps [startMoving, activate), then freezes.
//   • Deferred emission: JammerNoiseSource emits 300 mW continuous noise
//     only at step >= activate (wired via activateJammer() call in mobility).
//
// Radio choice: NoiseSource registers directly as IRadio and pushes noise
// transmissions to the medium without any Mac/CSMA layer. This is the
// simplest INET mechanism for a non-deferring continuous interferer.
//
package rlsmoke;

import inet.physicallayer.wireless.noise.NoiseSource;

module JammerHost extends NoiseSource
{
    parameters:
        @class(inet::JammerNoiseSource);
        @display("i=device/accesspoint_s;r=,,#FF0000");

        // Override NoiseSource's StationaryMobility with gradient-seeking mobility.
        // Pattern mirrors DroneHost / UserHost in SmokeNet.ned.
        mobility.typename = default("JammerGradientMobility");
}
