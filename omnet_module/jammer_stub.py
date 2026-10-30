#!/usr/bin/env python3
"""
Minimal ZMQ REP stub for jammer_verify config.

Returns zero-velocity actions for each drone and exits cleanly when
OMNeT++ sends done=true. No traffic assertions — the jammer is supposed
to disrupt delivery, so smoke_test.py's BONN CHECK 3 would always fail.
"""
import argparse
import json
import zmq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--num-drones", type=int, default=5)
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[jammer_stub] bound REP on tcp://127.0.0.1:{args.port}", flush=True)

    while True:
        msg = sock.recv()
        obs = json.loads(msg.decode("utf-8"))
        done = obs.get("done", False)
        step = obs.get("step", -1)

        if done:
            sock.send_string(json.dumps({"type": "ack"}))
            print(f"[jammer_stub] done at step {step}, exiting", flush=True)
            break

        actions = [[0.0, 0.0] for _ in range(args.num_drones)]
        sock.send_string(json.dumps({"type": "action", "actions": actions, "reward": 0.0}))

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
