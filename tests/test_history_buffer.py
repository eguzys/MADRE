"""Self-consistency regression test for PerDrone_OW / RawUserSet_OW's
K-deep history cache (ObservationWrappers.py, observation.py).

Background: the oldest slot (k=K-1) of the history ring buffer can hold an
older frame than "k steps ago". Direct tracing shows the buffer is
byte-exact; the older frame comes from `state_loss_rate` (default 0.3) and/or a physically dropped
link causing a peer to go unreached for one or more steps. When a peer
isn't reached, its cache slot is *correctly* left unshifted (see the
`aoi` docstring on `PerDrone_OW.__init__`: a slot holds "the age of the
observation stored at that slot -- NOT the slot index k"). Comparing slot
k against "the global current frame from k steps ago" is only a valid
invariant while every peer is reached on every step.

So this test pins `state_loss_rate=0.0` and uses a seed/N confirmed (by
directly tracing `nx.ancestors(graph, ...)` per step) to keep all nodes
mutually reachable for the whole run -- the one condition under which
"slot k == current frame from k steps ago" is guaranteed. Under those
conditions the invariant must hold exactly (no tolerance) for every
depth 0..K-1, on both wrapper classes that implement this cache.
"""
import numpy as np
import pytest

from MANET.build import build_padre_env
from MANET.Parallel_enviroments import PADREParallelEnv

_SEED = 123
_N = 4


def _collect_history(obs_wrapper, K, extra_kwargs, steps):
    env = PADREParallelEnv(lambda: build_padre_env(
        numbOfNodes=_N, numbOfUsers=(2, 20), seed=_SEED,
        obs_wrapper=obs_wrapper, history_K=K,
        fading_sigma_dB=0.0, state_loss_rate=0.0,
        **extra_kwargs,
    ))
    aoi_col = env._wrapper._aoi_col

    def newest(o):
        return np.asarray(o["drone_0"]).reshape(_N, K, -1)[:, 0].copy()

    obs, _ = env.reset()
    hist = [newest(obs)]
    for _ in range(steps):
        actions = {ag: env.action_space(ag).sample() for ag in env.agents}
        obs, _reward, _term, _trunc, _info = env.step(actions)
        hist.append(newest(obs))

    buf = np.asarray(obs["drone_0"]).reshape(_N, K, -1)
    return buf, hist, aoi_col


@pytest.mark.parametrize("obs_wrapper,extra_kwargs", [
    ("raw", {"encoder_mode": "learned"}),
    ("perdrone", {}),
])
@pytest.mark.parametrize("K", [1, 2, 4, 8, 16])
def test_history_slot_matches_current_frame_k_steps_ago(obs_wrapper, extra_kwargs, K):
    steps = K + 5
    buf, hist, aoi_col = _collect_history(obs_wrapper, K, extra_kwargs, steps)
    token_dim = buf.shape[-1]
    keep = [c for c in range(token_dim) if c != aoi_col]

    for k in range(K):
        delta = float(np.abs(buf[:, k][:, keep] - hist[-1 - k][:, keep]).sum())
        assert delta < 1e-6, (
            f"obs_wrapper={obs_wrapper} K={K} slot-{k} vs current[t-{k}]: "
            f"|delta|={delta:.6f} (excl. AoI column {aoi_col})"
        )


def test_k1_output_shape_unchanged():
    """history_K=1 must keep the pre-history (N, token_dim) per-drone
    output (no K axis folded into the width)."""
    env = PADREParallelEnv(lambda: build_padre_env(
        numbOfNodes=_N, numbOfUsers=(2, 20), seed=_SEED,
        obs_wrapper="raw", encoder_mode="learned",
        history_K=1, fading_sigma_dB=0.0, state_loss_rate=0.0,
    ))
    obs, _ = env.reset()
    drone0 = np.asarray(obs["drone_0"])
    assert drone0.shape == (_N, env._wrapper._cache_width + (1 if env._wrapper.mask else 0))
