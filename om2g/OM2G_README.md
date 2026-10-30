# OM2G — the narrowband ("2G") message-compression track

```
╔══════════════════════════════════════════════════════════════════════════╗
║  ⚠️  THIS IS **NOT** THE OMNeT TRACK.  ⚠️                                 ║
║                                                                          ║
║  Two separate experiments live in this repository.  Their numbers are    ║
║  NEVER comparable — different PHY, different channel capacity, different ║
║  reward scale, different observation semantics.                          ║
║                                                                          ║
║    track = OMNET   →  802.11g, 20 MHz, ~50 Mbps.  code/omnet_module/,    ║
║                       code/omnet_smoke/omnetpp.ini, network SmokeNet.    ║
║                       Thesis question: "does a permutation-invariant     ║
║                       aggregator beat an FFN?"                           ║
║                                                                          ║
║    track = OM2G    →  IEEE 802.15.4, 2.8 MHz, 250 kbps.  code/om2g/,     ║
║                       code/omnet_smoke/om2g.ini, network OM2GNet.        ║
║                       Thesis question: "is the encoder a good learned    ║
║                       MESSAGE COMPRESSOR for inter-agent state?"         ║
║                                                                          ║
║  Every OM2G artefact carries the literal token `OM2G` in its filename,   ║
║  its config-section name, its run name, its wandb tag, and a `track`     ║
║  column in every CSV it writes.  Nothing in the OMNET track contains     ║
║  that token.  If you are holding a number and cannot see `OM2G` on it,   ║
║  it is an OMNET number.                                                  ║
║                                                                          ║
║  The code enforces this: `om2g_track.require_om2g()` refuses to run      ║
║  unless OM2G_TRACK=1 is exported, and omnet_module/omnet_env.py refuses  ║
║  to run WITH it set.  You cannot mix them by accident.                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## What the track is for

The encoder is re-framed as a **learned compressor for the inter-agent state
message**. Each drone broadcasts its own state (position, noise floor, its
sensed user set, its MAC/PHY event summary) to its peers. Two arms:

| arm | what rides the wire | floats | UDP payload | 802.15.4 frames |
|---|---|---|---|---|
| `OM2G_COMPRESSED`   | encoder latent (D=8) | 22  | 88 B   | **1** |
| `OM2G_UNCOMPRESSED` | raw user block padded to `MAX_USERS_CAP=25` | 339 | 1356 B | **12** |

802.15.4 MTU is 127 − 9 = 118 B, so a UDP payload of ≤ 90 B fits one frame.
D=8 lands at 88 B: the compressed message is a **single frame**, the
uncompressed one is a **12-fragment IPv4 datagram**. That discrete cliff is
the headline.

## Files

| file | role |
|---|---|
| `om2g_track.py` | the guard + every OM2G constant (radio, message sizes). Single source of truth. |
| `om2g_env.py` | `OM2GEnv` — subclasses `OMNeTEnv`, adds delivery-gated peer messages + control-plane accounting |
| `om2g_regime_probe.py` | §8 gate: measures mean neighbour count and control-plane share |
| `om2g_task_gate.py` | re-runs the frozen C1–C6 task-validation criteria at the OM2G radio |
| `../omnet_smoke/om2g.ini` | the simulator config (`[Config OM2G_*]`) |
| `../omnet_smoke/rlsmoke/OM2GNet.ned` | the network (802.15.4 medium) |
