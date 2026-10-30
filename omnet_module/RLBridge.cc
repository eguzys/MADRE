//
// RLBridge.cc
//
#include "RLBridge.h"
#include "RLControlledMobility.h"
#include "inet/mobility/base/MobilityBase.h"

#include <cmath>
#include <sstream>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

namespace inet {

using json = nlohmann::json;

Define_Module(RLBridge);

RLBridge::RLBridge() = default;

RLBridge::~RLBridge()
{
    cancelAndDelete(syncEvent_);
    // sock_ and ctx_ unique_ptrs clean themselves up in correct order
    // (socket first, then context — unique_ptr destruction is reverse-of-decl).
}

void RLBridge::initialize()
{
    endpoint_     = par("pythonEndpoint").stringValue();
    syncInterval_ = par("syncInterval");
    numDrones_    = par("numDrones");
    maxSteps_     = par("maxSteps");

    // Size action buffer to numDrones, fill with NaN sentinel.
    actionBuffer_.assign(numDrones_,
                         { std::nan(""), std::nan("") });
    drones_.assign(numDrones_, nullptr);

    // ZMQ setup. REQ socket: we send, we block on reply.
    ctx_  = std::make_unique<zmq::context_t>(1);
    sock_ = std::make_unique<zmq::socket_t>(*ctx_, zmq::socket_type::req);
    sock_->connect(endpoint_);

    EV_INFO << "RLBridge: connected to " << endpoint_
            << ", numDrones=" << numDrones_
            << ", syncInterval=" << syncInterval_ << endl;

    // Schedule first sync at t=0. We send an initial observation so Python
    // has obs[0] to feed the policy and reply with action[0].
    syncEvent_ = new cMessage("rlSync");
    scheduleAt(SimTime(0), syncEvent_);
}

void RLBridge::registerDrone(int droneId, cModule* mobilityModule)
{
    if (droneId < 0 || droneId >= numDrones_)
        throw cRuntimeError("RLBridge: droneId %d out of range [0,%d)",
                            droneId, numDrones_);
    drones_[droneId] = mobilityModule;
}

void RLBridge::getAction(int droneId, double& outDx, double& outDy) const
{
    const auto& a = actionBuffer_.at(droneId);
    outDx = a.first;
    outDy = a.second;
}

void RLBridge::handleMessage(cMessage* msg)
{
    if (msg != syncEvent_)
        throw cRuntimeError("RLBridge got unexpected message");

    bool done = (stepCounter_ >= maxSteps_);

    // 1. Build observation JSON
    std::string obsJson = buildObservationJson(done);

    // 2. Send to Python (blocking)
    zmq::message_t req(obsJson.data(), obsJson.size());
    sock_->send(req, zmq::send_flags::none);

    EV_DEBUG << "RLBridge: sent obs step=" << stepCounter_
             << " done=" << done << endl;

    // 3. Block on reply
    zmq::message_t rep;
    auto result = sock_->recv(rep, zmq::recv_flags::none);
    if (!result)
        throw cRuntimeError("RLBridge: recv returned no message");

    std::string actionJson(static_cast<const char*>(rep.data()), rep.size());
    EV_DEBUG << "RLBridge: got actions of " << rep.size() << " bytes" << endl;

    // 4. Parse and store actions for drones to read
    parseActionJson(actionJson);

    stepCounter_++;

    // 5. Reschedule unless episode is over
    if (!done) {
        scheduleAt(simTime() + syncInterval_, syncEvent_);
    }
}

std::string RLBridge::buildObservationJson(bool done)
{
    json j;
    j["type"] = "obs";
    j["step"] = stepCounter_;
    j["done"] = done;

    json positions = json::array();
    for (int i = 0; i < numDrones_; ++i) {
        cModule* m = drones_[i];
        if (!m) {
            // Drone hasn't registered yet — should never happen after init.
            positions.push_back({0.0, 0.0});
            continue;
        }
        auto* mob = check_and_cast<MobilityBase*>(m);
        Coord p = mob->getCurrentPosition();
        positions.push_back({p.x, p.y});
    }
    j["positions"] = positions;

    // Smoke test: no real throughput yet. Constant 0.
    j["throughput"] = 0.0;

    return j.dump();
}

void RLBridge::parseActionJson(const std::string& s)
{
    json j = json::parse(s);
    if (!j.contains("actions"))
        throw cRuntimeError("RLBridge: action JSON missing 'actions' key");

    auto actions = j["actions"];
    if (!actions.is_array() || (int)actions.size() != numDrones_)
        throw cRuntimeError("RLBridge: expected %d actions, got %d",
                            numDrones_, (int)actions.size());

    for (int i = 0; i < numDrones_; ++i) {
        auto a = actions[i];
        if (!a.is_array() || a.size() != 2)
            throw cRuntimeError("RLBridge: action %d malformed", i);
        actionBuffer_[i] = { a[0].get<double>(), a[1].get<double>() };
    }
}

void RLBridge::finish()
{
    // Tell Python we're done (best-effort; if Python already closed, ignore).
    try {
        json j;
        j["type"] = "obs";
        j["step"] = stepCounter_;
        j["done"] = true;
        j["positions"] = json::array();
        j["throughput"] = 0.0;
        std::string s = j.dump();
        zmq::message_t m(s.data(), s.size());
        sock_->send(m, zmq::send_flags::dontwait);
    }
    catch (...) {
        // swallow — we're shutting down
    }
}

} // namespace inet
