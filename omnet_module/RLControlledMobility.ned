//
// RLControlledMobility — one instance per drone. Reads its velocity from
// the shared RLBridge module each time move() is called. Does not do
// any I/O itself.
//
package inet.mobility.single;

import inet.mobility.base.MovingMobilityBase;

simple RLControlledMobility extends MovingMobilityBase
{
    parameters:
        @class(RLControlledMobility);
        @display("i=block/control");

        // Numeric id of this drone. 0..numDrones-1. Must be unique and match
        // the order the Python side expects.
        int droneId;

        // Absolute path to the RLBridge module in the network. Default
        // assumes it lives at the top level as `rlBridge`.
        string bridgeModule = default("^.rlBridge");

        // Cap on velocity magnitude per sim-second. If Python sends a
        // (dx, dy) with |v| > this, we clip it. Keeps trained policies'
        // motion bounded under varying action scales.
        double maxSpeed @unit(mps) = default(10mps);

        // How often this module updates its visible position. Should be
        // <= bridge.syncInterval. 0.1s = 10x smoother visuals than the
        // RL decision rate.
        updateInterval @unit(s) = default(0.1s);
}
