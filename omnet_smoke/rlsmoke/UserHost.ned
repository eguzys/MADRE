//
// UserHost — ground user for Bonn Disaster Area scenario.
//
// Stub leaf node: does NOT participate in AODV. Routing is performed entirely
// by the drone backbone (drones run AODV among themselves). Users have a single
// default route pointing at their associated drone, installed by UserAssociationModule.
// Uses RLControlledMobility in user mode: position is sampled per-episode by
// RLBridge at INITSTAGE_TRANSPORT_LAYER; users are stationary within an episode.
//
// Apps configured in ini:
//   app[0] = UserAssociationModule  (BRP association + default-route management)
//   app[1] = UserBonnTrafficApp     (Bonn-hierarchy CBR traffic)
//
package rlsmoke;

import inet.node.inet.AdhocHost;

module UserHost extends AdhocHost
{
    parameters:
        int userId;
        string role = default("field");   // "command" | "section" | "field"
        int incidentId = default(-1);     // -1 = command-zone head

        // 802.11g, 2.4 GHz, 10 mW
        wlan[*].opMode = default("g(erp)");
        wlan[*].radio.typename = default("Ieee80211ScalarRadio");
        wlan[*].radio.bandName = default("2.4 GHz");
        wlan[*].radio.transmitter.power = default(10mW);

        // Per-episode randomized position via RLBridge sampling.
        mobility.typename     = default("RLControlledMobility");
        mobility.droneId      = default(-1);
        mobility.userId       = default(this.userId);
        mobility.bridgeModule = default("^.^.rlBridge");

        @display("i=device/cellphone_s;r=,,#CC3333");
}
