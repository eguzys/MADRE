"""BenchMARL TaskClass + Task enum for the OMNeT++ drone-swarm environment.

Mirrors common.py / PadreClass exactly, but drives OMNeTEnv instead of
PADREParallelEnv.  The PADRE path is not touched.
"""

import copy
import os
import sys
from typing import Callable, Dict, List, Optional

# ── sys.path injection so omnet_module (no __init__.py) is importable ──────
_OMNET_MODULE = os.path.join(os.path.dirname(__file__), "..", "..", "omnet_module")
sys.path.insert(0, os.path.abspath(_OMNET_MODULE))
from omnet_env import OMNeTEnv  # noqa: E402

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooWrapper, TransformedEnv
from torchrl.envs.transforms import RewardScaling

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

# ── Reward scale ─────────────────────────────────────────────────────────────
#
# A single constant affine value-target scale (instead of a running PopArt
# normalizer), applied here at collection time -- upstream of GAE
# (benchmarl Mappo.process_batch -> experiment.py's training loop consumes
# the collector's already-scaled reward). Per-step oracle reward is ~125
# over ~1000-step episodes, so GAE value targets are O(1e4-1e5); scaling by
# 1e-4 brings them to O(1-10) -- see
# src/tests/test_bug24_critic_value_normalization.py for the assert that
# checks this empirically against a real batch. Defensible in the paper as
# "rewards scaled by a constant for numerical conditioning."
#
# Applied via a TorchRL RewardScaling transform, not inside OMNeTEnv.step(),
# so the env keeps emitting the true raw oracle reward: evaluate_omnet.py,
# REWARD.md's documented examples, and the LP-correctness diagnostics
# (valley_probe.py, phase2_delivered.py, test_cache_parity.py) all call
# compute_oracle_max_flow_reward / env.step() directly and must keep seeing
# real units. Only the BenchMARL/TorchRL training-and-eval collection path
# (this file) sees the scaled value.
REWARD_SCALE = 1e-4


class _EnvFnList(list):
    """A list of env-factories that is also callable.

    BenchMARL calls get_env_fun(...)() to build test_env, and passes the same
    return value to ParallelEnv(N, fn) for training. Being both satisfies both
    without patching BenchMARL. Calling it uses worker 0.
    """

    def __call__(self):
        assert len(self), "_EnvFnList is empty -- num_envs was 0"
        return self[0]()


class OMNeTClass(TaskClass):
    """BenchMARL TaskClass for the OMNeT++ drone-swarm environment.

    Wraps OMNeTEnv with PettingZooWrapper so TorchRL can drive it.
    Same PettingZoo contract as PADREParallelEnv → spec methods copied
    identically from PadreClass.
    """

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        def _make_one(worker_index: int):
            def _make_env():
                # encoder_mode comes from the task yaml (omnet/static:
                # "frozen", omnet/static_learnedenc: "learned") -- see
                # conf/task/omnet/static*.yaml. Other OMNeT params are still
                # module constants in omnet_env.py; thread more of
                # self.config through as those become yaml-controlled too.
                encoder_mode = self.config.get("encoder_mode", "frozen")
                assert encoder_mode in ("frozen", "learned"), (
                    f"task config encoder_mode must be 'frozen' or "
                    f"'learned', got {encoder_mode!r} -- check "
                    f"conf/task/omnet/*.yaml for a typo."
                )
                # ── Track selection (OM2G) ───────────────────────────
                # $OM2G_TRACK=1 swaps in OM2GEnv, which is OMNeTEnv plus the
                # delivery gating that forces each drone's state message onto
                # the simulated 802.15.4 radio (code/om2g/om2g_env.py). It is
                # imported LAZILY, inside the guard, so the OMNET track never
                # loads the OM2G module and cannot be affected by it.
                #
                # om2g_env.require_om2g() and omnet_env._launch_omnet's ini
                # guard both re-check the same variable, so a half-configured
                # run fails at construction rather than producing an OM2G
                # policy scored in an 802.11g world.
                if os.environ.get("OM2G_TRACK") == "1":
                    import sys as _sys
                    _om2g_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__)))), "om2g")
                    if _om2g_dir not in _sys.path:
                        _sys.path.insert(0, _om2g_dir)
                    from om2g_env import OM2GEnv  # noqa: E402
                    pz_env = OM2GEnv(worker_index=worker_index,
                                     encoder_mode=encoder_mode)
                else:
                    pz_env = OMNeTEnv(worker_index=worker_index,
                                      encoder_mode=encoder_mode)
                wrapped = PettingZooWrapper(
                    env=pz_env,
                    return_state=True,
                    use_mask=True,
                    categorical_actions=False,
                    seed=seed,
                    device=device,
                )
                # Read the group key from the wrapper itself (drone_0..drone_N
                # -> group "drone") rather than hardcoding it -- never hardcode
                # a shape/key derivable from the spec (same convention as
                # architectures/gat_model.py's token_dim read).
                reward_keys = [(group, "reward") for group in wrapped.group_map]
                return TransformedEnv(
                    wrapped,
                    RewardScaling(
                        loc=0.0, scale=REWARD_SCALE,
                        in_keys=reward_keys, out_keys=reward_keys,
                    ),
                )
            return _make_env

        # TorchRL broadcasts a SINGLE callable into [fn] * N (batched_envs.py:424)
        # -- N references to one object, so a per-worker index cannot be baked in.
        # A Sequence is accepted instead (batched_envs.py:188) and indexed per
        # worker at :1635, pickled independently into each process. BenchMARL
        # also CALLS this return value directly for test_env
        # (experiment.py:448), so it must be both callable and indexable.
        return _EnvFnList([_make_one(i) for i in range(num_envs)])

    # --- capability flags ---

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("ep_length", 500)

    # --- spec / shape methods (identical to PadreClass — same PZ contract) ---

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return "omnet"


class OMNeTTask(Task):
    """Enum of available OMNeT++ tasks."""

    STATIC = None

    @staticmethod
    def associated_class():
        return OMNeTClass
