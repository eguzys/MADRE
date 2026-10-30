"""BenchMARL TaskClass + Task enum for the PADRE anti-jamming environment.

This file is the bridge between BenchMARL and our PettingZoo env.
- PadreClass: tells BenchMARL HOW to build the env and WHAT shapes to expect.
- PadreTask: enum of available PADRE tasks (just ANTI_JAMMING for now).
"""

import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooWrapper

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

# Your existing env builder
# Import from MANET folder
from MANET.build import build_padre_env
from MANET.Parallel_enviroments import PADREParallelEnv


class PadreClass(TaskClass):
    """BenchMARL TaskClass for PADRE.

    Wraps PADREParallelEnv (built via build_padre_env) with PettingZooWrapper
    so TorchRL can drive it.
    """

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        # Inject seed into config so build_padre_env gets it
        config["seed"] = seed

        def _make_env():
            # build_padre_env returns the inner PerDrone_OW.
            # Wrap with PADREParallelEnv to get the PettingZoo ParallelEnv interface.
            pz_env = PADREParallelEnv(build_env_fn=lambda: build_padre_env(**config))
            return PettingZooWrapper(
                env=pz_env,
                return_state=True,        # exposes state() for centralized critic
                use_mask=True,            # variable-N — Phase 2
                categorical_actions=False, 
                seed=seed,
                device=device,
            )

        return _make_env

    # --- capability flags ---

    def supports_continuous_actions(self) -> bool:
        # TODO the action space is 2 numbers likely corespoinding to dx dy. which makes unilkely to pick 0 movement under gaussian... 
        # we can add a threshold to treat small movements as zero
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False  # render is handled separately via PADRE's plot mode

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("ep_length", 1000)

    # --- spec / shape methods ---

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
        # this is the observation of the critic. 
        # TODO we can make the critic even more powerfull by giving more then the dreone info... like all the jammer and user lockations
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None  # no action masking in PADRE (all actions always legal)

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None  # no per-step info dict exposed

    @staticmethod
    def env_name() -> str:
        return "padre"


class PadreTask(Task):
    """Enum of available PADRE tasks."""

    STATIC = None

    @staticmethod
    def associated_class():
        return PadreClass