//
// UserAssociationModule — position-based BRP association for user nodes.
//
// Reads drone positions from RLBridge every evalInterval simulated seconds,
// computes log-distance-pathloss RSSI (alpha=2.5, DRONE_TX=200 mW),
// applies BRP score = RSSI - beta * user_count, and switches default route
// when best candidate beats current by >= hysteresis dB.
//
// v1 simplification: position-based RSSI rather than sniffing actual radio
// packets (UDP broadcast routing conflicts with AODV's empty-routing-table
// setup). See compile_users.md for full rationale.
//
package rlsmoke;

import inet.applications.contract.IApp;

simple UserAssociationModule like IApp
{
    parameters:
        @class(inet::UserAssociationModule);
        int    userId;
        string bridgeModule  = default("^.^.rlBridge");
        double beta          = default(3.0);    // dB per associated user
        double hysteresis    = default(3.0);    // dB
        double evalInterval  @unit(s) = default(1s);

    gates:
        input  socketIn;
        output socketOut;
}
