//
// RLControlledMobility — shared mobility module for drones and stationary users.
//
// Drone mode (droneId >= 0):
//   Reads velocity (dx,dy) from RLBridge each move() call. Velocity clipped
//   to maxSpeed. NaN velocity = hold position (before first Python action).
//
// User mode (userId >= 0):
//   Holds the position sampled by RLBridge at episode init (per-episode
//   randomization). Users are stationary within an episode (v1).
//
package rlsmoke;

import inet.mobility.base.MovingMobilityBase;

simple RLControlledMobility extends MovingMobilityBase
{
    parameters:
        @class(inet::RLControlledMobility);
        @display("i=block/control");

        // Drone id. -1 when used for users.
        int droneId = default(-1);

        // User id. -1 when used for drones.
        int userId = default(-1);

        // Absolute path to the RLBridge module in the network.
        string bridgeModule = default("^.^.rlBridge");

        // Cap on drone velocity magnitude per sim-second. Unused in user mode.
        double maxSpeed @unit(mps) = default(10mps);

        // How often this module updates its visible position.
        updateInterval @unit(s) = default(0.1s);
}
