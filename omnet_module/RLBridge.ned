//
// RLBridge — single point of contact between the OMNeT++ simulation and a
// Python RL process. Holds one ZMQ REQ socket. Once per syncInterval, sends
// a JSON observation to Python and blocks on the reply containing actions
// for all drones. Each RLControlledMobility module looks up this bridge at
// init time and reads its own velocity from it on every move() call.
//
package inet.mobility.single;

simple RLBridge
{
    parameters:
        @display("i=block/cogwheel");

        // Endpoint Python is bound to. Default loopback REQ→REP.
        string pythonEndpoint = default("tcp://127.0.0.1:5555");

        // Simulation time between RL decision points.
        double syncInterval @unit(s) = default(1s);

        // Number of drones. Must match the number of RLControlledMobility
        // instances in the network. Used to size the action/obs vectors.
        int numDrones;

        // Episode length in syncInterval units; when stepCounter reaches
        // this, we send done=true to Python and stop scheduling self-events.
        int maxSteps = default(1000);
}
