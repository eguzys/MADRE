// UserBonnTrafficApp — Bonn talk-group voice traffic application.
//
// Emits fixed-size MELPE frames (21 bytes × packetSizeMultiplier) to IPv4
// multicast addresses under direction of the RLBridge conversation process.
// Parameters mirror Aschenbruck IPCCC 2007 + LCN 2006 constants exactly.
//
package rlsmoke;

import inet.applications.contract.IApp;

simple UserBonnTrafficApp like IApp
{
    parameters:
        @class(inet::UserBonnTrafficApp);
        int    userId;
        string bridgeModule = default("^.^.rlBridge");
        int    listenPort   = default(5001);

        // ── Conversation-process probabilities (Aschenbruck IPCCC 2007 Table 2) ──
        double headInitiatorProb   = default(0.46);   // P(head initiates)
        double meshProb            = default(0.136);  // P(mesh call | member initiates)
        double transHeadToHead     = default(0.142);  // P(head speaks again | head just spoke)
        double transMemberToMember = default(0.190);  // P(member speaks again | member just spoke)
        double callContinuationProb= default(0.7);    // P(another call in same conversation)

        // ── Holding-time distributions (Aschenbruck LCN 2006) ────────────────────
        double callCHT_meanlog  = default(0.4438584); // lognormal μ for call holding time [s]
        double callCHT_sdlog    = default(0.7766512); // lognormal σ
        double callIdle_shape   = default(1.530391);  // gamma α for inter-call idle [s]
        double callIdle_rate    = default(1.606532);  // gamma rate (1/scale)
        double convIdle_meanlog = default(1.390844);  // lognormal μ for conv-idle [s]
        double convIdle_sdlog   = default(1.628825);  // lognormal σ
        double convIdle_shift   = default(3.0);       // shift: T = sample + 3 s

        // ── MELPE codec (G.723.1-like 2.4 kbps encoding) ─────────────────────────
        int    melpePayloadBytes   = default(21);     // bytes per frame
        double melpeInterval_ms    = default(67.5);   // ms between frames
        int    packetSizeMultiplier= default(1);      // scale payload

    gates:
        input  socketIn;
        output socketOut;
}
