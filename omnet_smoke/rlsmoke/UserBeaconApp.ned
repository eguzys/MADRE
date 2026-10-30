//
// UserBeaconApp -- periodic PRESENCE beacon on each ground user.
//
// A user that is not currently the speaker of a conversation transmits
// nothing at all on the OM2G track, so a drone cannot sense it: the drone's
// only sensing channel is the received frame. This app puts a small periodic
// multicast frame on the medium from EVERY user, calling or not, so presence
// becomes observable. The drone side needs no change -- RLBridge's radio-level
// packetSentToUpper hook already sets user_rssi/user_rssi_valid for any
// completed reception whose source MAC maps to a user.
//
// Multicast, not broadcast, for the same reason as DroneBeaconApp: broadcast
// needs a routing table these nodes deliberately do not have. The app pins its
// egress interface with the MULTICAST_IF socket option, so it does not depend
// on UserAssociationModule's default route either.
//
// Port and group are deliberately DISTINCT from the drone beacon's
// 5002 / 239.1.2.1, so user presence traffic can never be mistaken for, or
// counted as, drone control traffic.
//
// This app is OFF BY DEFAULT: it is only instantiated when $OM2G_USER_BEACON=1
// makes om2g_config.om2g_extra_args() emit the *.user[*].app[2] assignments.
//
package rlsmoke;

import inet.applications.contract.IApp;

simple UserBeaconApp like IApp
{
    parameters:
        @class(inet::UserBeaconApp);
        // -1 (the default) means "ask the containing UserHost for its userId",
        // so the app can be wired from the command line without a
        // `parent.userId` NED expression having to survive shell escaping.
        int    userId = default(-1);
        double beaconInterval @unit(s) = default(1s);
        int    beaconPort = default(5003);
        string beaconGroup = default("239.1.2.2");
        // Small on purpose: this frame carries presence, not state. What must
        // reach the drone is the FRAME, not its contents.
        int    beaconBytes = default(1);
        // Per-beacon stdout line ("USERBEACON:<t>,<userId>,<bytes>"). Off by
        // default -- at 40 users x 1 Hz this is otherwise a lot of log.
        bool   verbose = default(false);

    gates:
        input  socketIn;
        output socketOut;
}
