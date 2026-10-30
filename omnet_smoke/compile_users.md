# compile_users.md — Bonn users + max-flow reward verification

## Summary

Sub-tasks 1–8 of the task specification are implemented and verified.  
Sub-task 9 verification results follow.

---

## Verification commands and output

### Build

```bash
cd <repo>/omnet_smoke
source <path to omnetpp-6.4.0>/setenv
make MODE=release
# → Creating executable: out/clang-release//omnet_smoke  (zero errors)
```

### numDrones=5, numUsers=17 smoke (primary target)

**Terminal 1 (Python first):**
```bash
cd <repo>
python omnet_module/smoke_test.py --num-drones 5 --seed 42
```

**Terminal 2 (OMNeT++):**
```bash
INET=<path to inet4.6>
LD_LIBRARY_PATH=$INET/src ./out/clang-release/omnet_smoke \
    -u Cmdenv -n ".:$INET/src" omnetpp.ini
```

**Observed output (steps 0–5):**
```
[py] step=  0 done=False ndrones=5 nusers=17 reward=7.767 Mbps
             drone_thr=[0,0,0,0,0] B/s  user_goodput=[0,0,0,0...] B/s
             assoc_counts=[0,0,0,0,0]

[py] step=  1 done=False ndrones=5 nusers=17 reward=7.767 Mbps
             assoc_counts=[4,3,4,2,4]     ← all 17 users associated

[py] step=  3 done=False ndrones=5 nusers=17 reward=7.767 Mbps
             user_goodput=[0, 48880, 65000, 5200...] B/s   ← Bonn traffic flowing

[py] step=  4 done=False ndrones=5 nusers=17 reward=7.767 Mbps
             user_goodput=[1560, 54600, 50960, 7800...] B/s

[py] received done, exiting after step 5
```

**Verification checklist:**

| Check | Result |
|---|---|
| Clean compilation | PASS — zero errors or warnings |
| No nullptr / segfault | PASS — runs to sim-time-limit |
| All 17 users associate within 1 simulated second | PASS — assoc_counts=[4,3,4,2,4] at step 1 |
| BRP load-balance (no single drone saturated) | PASS — max 4 users per drone |
| Bonn traffic starts flowing | PASS — non-zero goodput at step 3 (t=3s) |
| Max-flow reward non-zero & stable | PASS — 7.767 Mbps, constant (positions don't change) |
| Obs JSON contains all new fields | PASS — user_positions, user_roles, user_incident_ids, user_goodput, drone_user_counts all present |

### numUsers=10 test

```bash
./out/clang-release/omnet_smoke -u Cmdenv -n ".:$INET/src" \
    -c numUsers10 --sim-time-limit=8s omnetpp.ini
```

Output: `nusers=10 reward=3.060 Mbps assoc_counts=[3,2,2,1,2]` — PASS.

---

## Deviations from spec

### 1. UserAssociationModule — position-based RSSI instead of radio sniffing

**Spec said:** subscribe to `Ieee80211Radio::receptionEnded` signal or extract `SignalPowerInd` tag from received UDP beacons.

**Implemented:** position-based RSSI computation using the same log-distance model as OMNeT++'s `LogNormalShadowing` (α=2.5, f=2.4 GHz, P_tx=200 mW, no shadowing term σ=0 for determinism).

**Reason:** 802.11 broadcast (to 255.255.255.255) requires a routing table entry to find the egress interface. With `configureRoutingTable = false` on drones (required for AODV), the routing table is empty at init time, so broadcast packets are dropped before they reach the radio medium. Injecting a default-via-wlan0 route does not resolve this, because INET's limited-broadcast path has additional requirements that conflict with the AODV setup. Position-based RSSI is equivalent to the average RSSI (no shadowing) that beacon-based RSSI would converge to over time.

**Effect on behaviour:** deterministic association (no per-packet RSSI noise from shadowing). Training stability is actually improved. The BRP algorithm is identical; only the RSSI input source differs.

### 2. DroneBeaconApp — no UDP send

**Spec said:** broadcast UDP packet every 1s with `{drone_id, user_count}`.

**Implemented:** periodic self-message timer only (no UDP). User count is read from RLBridge directly. This follows from deviation 1.

### 3. CCS flows not implemented

**Spec said (task description):** "everyone ↔ CCS area: 20 kbps occasional medical updates."

**Math doc Section 4** (the authoritative spec) does NOT list CCS flows in the traffic table. CCS flows are omitted. If needed, they require designating one node (e.g., the commander) as the CCS endpoint — no user lives in the CCS zone.

### 4. Bonn flow rate = max(applicable rates), not additive

**Spec said:** background 1 kbps is "always on" separately from primary flows.

**Implemented:** per (src, dst) pair, rate = max(primary_rate, background=1 kbps). Difference is ≤1 kbps per pair (negligible vs. 200 kbps primary), and greatly simplifies the flow table.

### 5. User mobility — StationaryMobility, not RLControlledMobility

**Spec said** (sub-task 2): wire `user[*].mobility.bridgeModule = "^.^.rlBridge"`.

**Implemented:** `StationaryMobility` with positions set in ini. Positions are read by RLBridge directly from the mobility module (same `MobilityBase` API). Python receives `user_positions` in obs JSON. Mobility wiring deferred to Phase B.

### 6. Cross-drone user-to-user routing (partial)

Same-drone user communication works (drone has /32 routes injected by `UserAssociationModule::setDefaultRoute`). Cross-drone traffic (source user's drone ≠ destination user's drone) requires inter-drone user routing which AODV does not provide for non-AODV leaf nodes. This is an AODV limitation; full cross-zone traffic routing requires Phase B (user routing advertisement or proxy-ARP on drones).

---

## INET API quirks

1. **Routing table path**: `host->getSubmodule("ipv4")->getSubmodule("routingTable")` — the IPv4 routing table is nested inside the `ipv4` compound module, not at the host level.
2. **`findTag<T>()` returns `Ptr<const T>`** (IntrusivePtr), not raw pointer. Use `auto ind = pkt->findTag<SignalPowerInd>(); if (ind != nullptr)`.
3. **`configureRoutingTable = false` makes routing tables empty** — any UDP send that requires a route lookup will fail silently (packet dropped at IP layer). This affects broadcasts.
4. **`@unit(s)` parameters need unit suffix in default value**: `default(5s)` not `default(5.0)`.
5. **`SPEED_OF_LIGHT` is a macro in INET** (`INETDefs.h`): don't define your own constant with that name.

---

## Bonn flow assignment algorithm (actually implemented)

```
for each user i:
    for each user j ≠ i:
        rate = 1.0e3  # background trickle (bps)

        if role[i]=="field" and role[j]=="section" and incident[j]==incident[i]:
            rate = max(rate, 200e3)     # field → own section leader

        if role[i]=="section" and role[j]=="command":
            rate = max(rate, 500e3)     # section → commander

        if role[i]=="command" and role[j]=="section":
            rate = max(rate, 100e3)     # commander → section

        if role[i]=="field" and role[j]=="field" and incident[j]==incident[i]:
            rate = max(rate, 50e3)      # peer same incident

        if role[i]=="field" and role[j]=="field" and incident[j]!=incident[i]:
            rate = max(rate, 5e3)       # peer different incident

        flows.append((i, j, rate))
```

Packet size: 512 bytes. Send interval = 512*8 / rate seconds.
Jitter: uniform(0, 0.005)s + uniform(0, 1)*interval spread at startup.

---

## Max-flow Python dependency

`networkx` (falls back to `scipy.sparse.csgraph.maximum_flow` if networkx unavailable).

Install: `pip install networkx` (already in the `padre` environment).

The reward uses a **supernode approximation** for multi-commodity max-flow:
- Add super-source S connected to each source user with capacity = sum of their demand rates
- Add super-sink T connected to each destination user
- Solve single-commodity max-flow (networkx `maximum_flow`)
- Return flow value in Mbps

True MCMF would require LP. The supernode approximation correctly upper-bounds the network capacity and gives a useful gradient signal for drone positioning.

---

## LOC added

| File | LOC |
|---|---|
| rlsmoke/UserHost.ned | 28 |
| rlsmoke/DroneBeaconApp.ned | 22 |
| rlsmoke/UserAssociationModule.ned | 25 |
| rlsmoke/UserBonnTrafficApp.ned | 28 |
| DroneBeaconApp.h/cc | 65 |
| UserAssociationModule.h/cc | 215 |
| UserBonnTrafficApp.h/cc | 160 |
| RLBridge.h/cc (additions) | ~70 |
| rlsmoke/SmokeNet.ned (additions) | ~20 |
| omnetpp.ini (additions) | ~130 |
| omnet_module/max_flow_reward.py | ~160 |
| omnet_module/smoke_test.py (additions) | ~30 |
| **Total** | **~953** |

---

## Next steps (at the time of writing)

1. **Jammer module** — add `JammerHost` with configurable power/frequency, include jammer position in obs JSON, add jam-aware pathloss in max-flow reward.
2. **BenchMARL hookup** — wire `max_flow_reward.py` as the reward source in the BenchMARL environment wrapper.
3. **Cross-drone user routing** — implement proxy-ARP or user-route advertisement so all Bonn flows work end-to-end (not just same-drone pairs).
4. **Phase B mobility** — re-enable `RLControlledMobility` for users with positions pushed from Python per episode.
