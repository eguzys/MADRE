//
// RLBridge — single point of contact between the OMNeT++ simulation and a
// Python RL process. Holds one ZMQ REQ socket. Once per syncInterval, sends
// a JSON observation to Python and blocks on the reply containing actions
// for all drones. Each RLControlledMobility module looks up this bridge at
// init time and reads its own velocity from it on every move() call.
//
// Routes: maintains /32 host routes for ALL users on ALL drones so
//         cross-drone user-to-user UDP delivery works.
// Episodes: samples user positions, roles, and zone centers per episode
//         at INITSTAGE_TRANSPORT_LAYER (before UserBonnTrafficApp reads roles).
//
package rlsmoke;

simple RLBridge
{
    parameters:
        bool dumpRoutes = default(false);  // print all routing tables at finish()
        @class(inet::RLBridge);
        @display("i=block/cogwheel");

        // Endpoint Python is bound to. Default loopback REQ→REP.
        string pythonEndpoint = default("tcp://127.0.0.1:5555");

        // Simulation time between RL decision points.
        double syncInterval @unit(s) = default(1s);

        // Number of drones. Must match the number of RLControlledMobility
        // instances in the network.
        int numDrones;

        // Number of user nodes (UserHost).
        int numUsers = default(0);

        // Episode length in syncInterval units.
        int maxSteps = default(1000);

        // Episode seed. 0 = use simulation RNG (varies per run).
        int episodeSeed = default(0);

        // Placement mode: "bonn" (default, disaster-zone sampling) or
        // "padre" (PADRE-style Gaussian clusters). Additive; bonn = byte-identical.
        string placementMode = default("bonn");
        // PADRE user Gaussian spread (m), std per axis. Faithful default = arena*0.05.
        double userSpread = default(30.0);   // BASE; scaled by arena/2000

        // Traffic mode: "bonn" (default, talk-group + multicast) or
        // "padre" (point-to-point unicast, one partner per sender).
        // Additive; bonn = byte-identical.
        string trafficMode @enum("bonn","padre") = default("padre");
        // PADRE traffic: probability a chosen partner is in a different cluster.
        double pCross = default(1.0);
        // PADRE traffic: per-user probability of being an active sender this
        // episode (PADRE's TRANSMITTER_PROB). At least one sender is guaranteed.
        double senderProb = default(1.0);

        // Arena dimensions (m).
        // Upper bound on the sampled cluster count. PADRE's geometric draw
        // (p=1/6) regularly yields 8-11 clusters; with >5 clusters, 5 drones are
        // structurally short of access drones AT ANY ARENA SIZE (coverage stuck
        // at 0.5-0.7). Capping at 3 gives 3 access + 2 relays, the target
        // topology. task_validation.py applies the same cap via --max-clusters.
        int    maxClusters      = default(3);

        // Drone spawn placement, applied AFTER the user clusters are sampled.
        //   "ini"     - keep initialX/initialY from omnetpp.ini (DEFAULT; every
        //               pre-existing config stays byte-identical)
        //   "cluster" - shuffled round-robin over cluster centroids + jitter.
        //               Every cluster covered; every drone starts on real users,
        //               so the reward gradient is nonzero at t=0.
        //   "staging" - all drones on the largest cluster (ablation).
        string droneSpawnMode   = default("ini");
        // userSpawnMode=="dumbbell": cluster centres are placed on a line with
        // pairwise separation pinned to dumbbellD (+/- dumbbellJitter) instead
        // of uniformly at random. This removes the separation lottery that makes the
        // coverage baselines bimodal (fine on close seeds, exactly 0 on far
        // ones). Users are Gaussian TRUNCATED at 2.5*sigma so the cluster
        // radius is a hard bound.
        string userSpawnMode    = default("padre");
        // droneSpawnMode=="explore": EXPLORING STARTS. With probability
        // exploreP the drones spawn uniformly at random instead of on the
        // clusters, so states containing a relay chain are visited during
        // TRAINING. Evaluation must use "cluster".
        double exploreP         = default(0.5);
        double dumbbellD        = default(2445.0);
        double dumbbellJitter   = default(75.0);
        // Upper edge of the cluster-separation BAND. dumbbellD is the minimum
        // pairwise separation; this bounds every cluster's NEAREST NEIGHBOUR
        // from above, so no episode contains only bridges too deep to learn.
        double maxNeighbourSep  = default(1750.0);
        double droneSpawnJitter = default(50.0);   // Gaussian sigma, metres
        // droneSpawnMode=="depot": every drone launches from one base,
        // spread on a small ring so co-located radios do not degenerate.
        double depotX    = default(60.0);
        double depotY    = default(60.0);
        double depotRing = default(25.0);

        // Arena dimensions (m).
        double arenaWidth  = default(2000.0);
        double arenaHeight = default(2000.0);

        // Zone radii (m) — from episode_parameter_math.md Section 3.
        double incidentZoneRadius = default(300.0);
        double commandZoneRadius  = default(100.0);  // updated to match defense paragraph
        double ccsZoneRadius      = default(100.0);  // per episode_parameter_math.md Section 3.5

        // Field users per incident bounds — Uniform{min..max}.
        int fieldPerIncidentMin = default(5);
        int fieldPerIncidentMax = default(9);

        // CCS staff per zone — Uniform{min..max} (BonnMotion 4 docs, Section 6.4).
        int ccsStaffPerZoneMin = default(2);
        int ccsStaffPerZoneMax = default(4);
}
