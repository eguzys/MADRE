//
// DroneBeaconApp -- periodic hello beacon on each drone.
//
// Sends a small UDP packet to a drone-only multicast group every
// beaconInterval, so peer drones physically receive the frame (RLBridge
// measures real peer RSSI from it at the radio hook) and AODV can discover
// multi-hop routes through the backbone.
//
// Multicast, not broadcast: broadcast requires a non-empty routing table,
// which conflicts with this sim's AODV setup. Multicast is the pattern
// UserBonnTrafficApp already uses successfully.
//
package rlsmoke;

import inet.applications.contract.IApp;

simple DroneBeaconApp like IApp
{
    parameters:
        @class(inet::DroneBeaconApp);
        int    droneId;
        string bridgeModule = default("^.^.rlBridge");
        double beaconInterval @unit(s) = default(1s);
        int    beaconPort = default(5002);
        string beaconGroup = default("239.1.2.1");
        int    beaconBytes = default(32);
        // Opt-in dynamic sizing: when true, beaconBytes above is IGNORED and
        // the payload is instead sized to this drone's LIVE sensed-user count
        // each beacon (beaconFixedBytes + sensedCount * beaconBytesPerUser),
        // never padded to OM2G_MAX_USERS_CAP. Default false reproduces every
        // existing run byte-identical. See DroneBeaconApp.cc.
        bool   dynamicBeaconSize = default(false);
        int    beaconFixedBytes = default(56);     // om2g_track.OM2G_MSG_FIXED_FLOATS * 4
        int    beaconBytesPerUser = default(52);   // om2g_track.OM2G_OBS_FEATURES * 4
        // Opt-in "cheap" send-gating heuristic: skip a beacon entirely
        // (no packet, no airtime) if this drone has moved less than
        // sendOnDeltaPosThresholdM AND its sensed-user count is unchanged
        // since the last beacon it actually sent. Default false reproduces
        // every existing run byte-identical.
        bool   sendOnDeltaEnabled = default(false);
        double sendOnDeltaPosThresholdM = default(10.0);
        // Opt-in "smart"/learned send-gate: the policy's own 3rd action
        // dimension (RLBridge::getLearnedGateWantsSend) decides send/skip
        // instead of the fixed sendOnDelta rule. Mutually meaningful with
        // sendOnDeltaEnabled off -- don't enable both at once, the learned
        // gate takes priority when both are set (see DroneBeaconApp.cc).
        bool   learnedGateEnabled = default(false);

    gates:
        input  socketIn;
        output socketOut;
}
