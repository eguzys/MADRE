from . import Environments as envs
from . import RewardWrappers as rw
from .ObservationWrappers import PerDrone_OW
from .observation import RawUserSet_OW
from .ThroughputLogger import ThroughputLogger
import os


def build_padre_env(
    numbOfNodes=4,
    numbOfJammers=1,
    numbOfUsers=(2, 20),
    seed=None,
    ep_length=1000,
    state_loss_rate=0.3,
    self_flag=True,
    aoi=True,
    history_K=1,
    mask=True,
    env_class="static",
    render_mode=None,
    obs_wrapper="perdrone",
    fading_sigma_dB=4.0,
    encoder_mode="frozen",
    max_users=None,
):
    """PADRE env + reward wrapper + PerDrone_OW for use inside PADREParallelEnv.

    Matches the construction used by train_perdrone.py, minus the
    PerDroneSingleStepEnv top layer (which exists only to emulate multi-agent
    behaviour inside SB3's single-agent contract).

    render_mode: pass "plot" to open the PADRE GUI; None (default) for
    headless operation.  Default None leaves training completely unaffected.

    max_users: only consulted by RawUserSet_OW's encoder_mode=="learned"
    branch. Overrides the padded user-block width it would otherwise infer
    from `numbOfUsers`, for callers (evaluation/runner.py) that must pass a
    scalar `numbOfUsers` to pin the live per-episode count while keeping the
    block sized to the training upper bound. None (default) leaves training
    completely unaffected -- it still infers the bound from `numbOfUsers`.
    """
    env_cls = envs.Dynamic_MANETEnv if env_class == "dynamic" else envs.Static_MANETEnv
    base = env_cls(
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        seed=seed,
        networkConnectedAtStart=False,
        render_mode=render_mode,
    )
    base.set_ep_length(ep_length)
    base.set_step_jammers_start_moving(100)
    base.set_steps_till_activate_jammers(500)

    rew = rw.LogReward(base)
    rew = ThroughputLogger(rew, enabled=os.environ.get("PADRE_DIAG") == "1")

    ow_cls = RawUserSet_OW if obs_wrapper == "raw" else PerDrone_OW
    ow_kwargs = dict(
        state_loss_rate=state_loss_rate,
        env=rew,
        self_flag=self_flag,
        aoi=aoi,
        history_K=history_K,
        mask=mask,
    )
    if obs_wrapper == "raw":
        ow_kwargs["fading_sigma_dB"] = fading_sigma_dB
        ow_kwargs["encoder_mode"] = encoder_mode
        ow_kwargs["max_users"] = max_users
    perdrone = ow_cls(**ow_kwargs)
    return perdrone