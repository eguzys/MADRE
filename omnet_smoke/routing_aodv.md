# Routing: AODV instead of static /32 routes

## Design

The simulation models a UAV swarm relaying MANET traffic between ground users. Drones and
ground users run AODV, so routes follow the drones as they move. Static per-association /32
routes are not used, for two reasons:

1. **Routes under mobility.** A /32 host route installed at association time points at the
   drone's position at that moment and has no rediscovery mechanism, so it would keep
   forwarding toward a next hop after the link to it has broken.

2. **Routing on user nodes.** A user built on `AdhocHost` (a bare IP host without AODV) would
   depend on a manually injected default route to its associated drone, refreshed only when the
   `UserAssociationModule` timer triggers a re-association, which is too coarse for the traffic.

## How OMNeT++/INET AODV Works

INET's AODV implementation (`inet.routing.aodv.Aodv`) runs as an application-layer module inside
every `AodvRouter` node. When a node needs to send a packet to a destination it has no route for,
AODV floods a Route Request (RREQ) message. Any node that has a route to the destination (or *is*
the destination) responds with a Route Reply (RREP) unicasted back along the reverse path. Routes
are soft-state: each entry carries an `activeRouteTimeout`; if no traffic flows to that destination
within that window, the route expires and the next packet triggers a fresh RREQ.

The `Ipv4NetworkConfigurator` normally pre-populates routing tables with static routes derived from
the network topology. For AODV, these static entries must be suppressed so AODV can manage the
table dynamically instead.

## Implementation

### 1. `rlsmoke/UserHost.ned`

`UserHost` `extends AodvRouter` rather than `AdhocHost`. This gives each ground user a
full AODV stack. Users participate in route discovery — they can respond to RREQ messages and
build reverse routes, which is essential for bidirectional traffic.

### 2. `omnetpp.ini` — configurator flags

Static routes are suppressed on the global configurator, following the INET-standard pattern used
in `inet4.6/examples/aodv/`:

```ini
*.configurator.addStaticRoutes  = false
*.configurator.addDefaultRoutes = false
*.configurator.addSubnetRoutes  = false
**.aodv.activeRouteTimeout = 1s
```

`activeRouteTimeout = 1s` is chosen because drones move at up to 20 m/s across a 2000 m arena.
AODV's default 3 s timeout would let routes persist well past the point where the link that
originally established them has broken. At 1 s the protocol rediscovers paths at roughly the same
cadence as drone movement, keeping the routing table current.

### 3. `RLBridge::notifyAllDronesOfAssociation`

The function does no IPv4Route manipulation. It only:
- Decrements `droneUserCounts_[oldDroneId]`
- Increments `droneUserCounts_[newDroneId]`
- Records `userToDrone_[userId]` and `userIps_[userId]`
- Logs the change

These three vectors feed the observation JSON sent to Python; that telemetry is unchanged.

### 4. `UserAssociationModule::setDefaultRoute`

The function installs no MANUAL routes. It only resolves `wlan0` to get `myIp` and calls
`notifyAllDronesOfAssociation` for the telemetry update. AODV handles the actual forwarding path.

The module holds no `routingTable_` reference and does not include `Ipv4Route.h` or
`IIpv4RoutingTable.h`.

## Unaffected parts

- `max_flow_reward.py`, zone sampling, talk groups, role assignment
- ZMQ bridge protocol (obs JSON shape, action JSON shape)
- `RLControlledMobility`, `UserBonnTrafficApp`, `DroneBeaconApp`
- The BRP score and hysteresis logic in `evaluateAssociation` — this is load-balancing telemetry
  exposed as `droneUserCounts_` in the obs JSON, not a routing mechanism
- Signal subscriptions and byte counters (`packetReceivedFromLower`, `packetSentToUpper`)
- `RLBridge::dumpAllRoutingTables` and the `dumpRoutes` ini parameter

## Smoke Verification

The key check from the verification spec is that `dumpAllRoutingTables` shows routes with
`src` values other than `0` (MANUAL). INET's `IRoute::SourceType` enum assigns AODV-installed
routes a non-zero value; once AODV has completed route discovery those entries will appear in the
dump. Expect a short delay at step 0 while AODV floods its initial RREQs; this is normal.

```bash
grep -rn "addRoute.*new Ipv4Route" omnet_smoke/
```

Should return no matches in `RLBridge.cc` or `UserAssociationModule.cc`.
