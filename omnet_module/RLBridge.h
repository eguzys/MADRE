//
// RLBridge.h — see RLBridge.ned for module-level docs.
//
#ifndef __INET_RLBRIDGE_H
#define __INET_RLBRIDGE_H

#include <vector>
#include <memory>

#include "inet/common/INETDefs.h"

// Forward-declare ZMQ types so the header doesn't pull in <zmq.hpp>.
namespace zmq { class context_t; class socket_t; }

namespace inet {

/**
 * Single-instance module. Holds the ZMQ REQ socket and the
 * latest-actions buffer that RLControlledMobility instances read from.
 *
 * Lifecycle:
 *   initialize(): open socket, schedule first sync event at t=0
 *   handleMessage(self-event): collect obs from all drones, send to Python,
 *                              block on reply, store actions, reschedule
 *   finish():     send done=true, close socket
 */
class INET_API RLBridge : public cSimpleModule
{
  public:
    RLBridge();
    virtual ~RLBridge();

    /// Called by RLControlledMobility to read its action. -1 = no action yet
    /// (we're before the first sync), drone should hold position.
    void getAction(int droneId, double& outDx, double& outDy) const;

    /// Called by RLControlledMobility at init time to register itself. The
    /// bridge needs to know the order in which to pack obs and actions.
    void registerDrone(int droneId, cModule* mobilityModule);

  protected:
    // OMNeT++ module API
    virtual void initialize() override;
    virtual void handleMessage(cMessage* msg) override;
    virtual void finish() override;

  private:
    // Pull observations from every registered drone and return JSON-ready data.
    // For the smoke test, obs = [(x, y), ...] for each drone.
    std::string buildObservationJson(bool done);

    // Parse the action reply from Python and fill actionBuffer_.
    void parseActionJson(const std::string& json);

    // ZMQ
    std::unique_ptr<zmq::context_t> ctx_;
    std::unique_ptr<zmq::socket_t> sock_;
    std::string endpoint_;

    // Sim
    SimTime syncInterval_;
    int numDrones_;
    int maxSteps_;
    int stepCounter_ = 0;

    // Self-message used to schedule sync points.
    cMessage* syncEvent_ = nullptr;

    // Registered drones, indexed by droneId. Filled in registerDrone().
    std::vector<cModule*> drones_;

    // Latest action per drone. Indexed by droneId. (NaN, NaN) means not-yet-set.
    std::vector<std::pair<double, double>> actionBuffer_;
};

} // namespace inet

#endif // __INET_RLBRIDGE_H
