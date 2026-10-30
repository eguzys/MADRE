"""
watch_agent.py — visually watch agents move over the PADRE field.

STAGE 1 (now): opens PADRE's 'plot' GUI and steps the env with dummy
actions, to confirm the visualization works.
STAGE 2 (later): load a trained BenchMARL checkpoint and drive the env
with the real policy instead of dummy actions.

Run:  python watch_agent.py
(Opens a GUI window. Run locally, not over headless SSH.)
"""

import sys
import numpy as np

sys.path.append(".")

from MANET.Parallel_enviroments import PADREParallelEnv
from MANET.build import build_padre_env


def get_base_env(parallel_env):
    """Walk down from PADREParallelEnv to the Static_MANETEnv that owns the
    'plot' GUI, unwrapping the obs/reward wrappers in between."""
    e = parallel_env._wrapper          # PerDrone_OW
    while hasattr(e, "env"):
        e = e.env                      # unwrap RewardWrapper -> Static_MANETEnv
    return e


def main(steps=300):
    env = PADREParallelEnv(build_env_fn=lambda: build_padre_env(numbOfUsers=10))

    # Turn on the 'plot' GUI on the underlying Static_MANETEnv.
    base = get_base_env(env)
    if hasattr(base, "setRenderMode"):
        base.setRenderMode("plot")
        print("render mode set to 'plot' on", type(base).__name__)
    else:
        print("WARNING: no setRenderMode on", type(base).__name__)

    obs, _ = env.reset()
    print("env reset, agents:", env.agents)

    for t in range(steps):
        # STAGE 1: dummy actions (random). STAGE 2 will replace this block
        # with a trained policy's action.
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, terms, truncs, _ = env.step(actions)
        base.render()  # draw this frame
        if any(terms.values()) or any(truncs.values()):
            print(f"episode ended at step {t}")
            break

    print("done")


if __name__ == "__main__":
    main()
