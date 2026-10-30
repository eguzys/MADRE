"""
ThroughputLogger — comprehensive diagnostic pass-through wrapper.

A single place for ALL training-time env diagnostics logged to wandb.
Pure observation: never modifies reward, obs, action, or env behaviour.

Toggle with the `enabled` constructor arg (default False -> total no-op,
zero overhead). Set enabled=True (e.g. gated on an env var in build.py) to
stream diagnostics into the active wandb run.

All keys prefixed 'diag/'. Grouped:

  PER-STEP
    throughput_raw          max-flow value (bits/s)
    throughput_mbps         throughput_raw / 1e6
    throughput_utilization  throughput / max_possible_throughput  (0..1)
    reward                  scalar reward this step
    reward_abs              |reward|
    action_norm_mean        mean L2 norm of the joint action (movement magnitude)

  PER-EPISODE (logged on terminated/truncated)
    ep_len
    ep_outcome              0=truncated, 1=terminated
    ep_reward_max / _min / _mean / _median / _std
    ep_max_median_ratio     dynamic-range indicator (the PPO-stability metric)
    ep_throughput_max / _mean
    ep_utilization_mean
    ep_reward_hist          distribution of per-step rewards this episode
    ep_throughput_hist      distribution of per-step throughput this episode
    ep_total_movement       summed action norm over the episode

  RUNNING (rolling window across episodes)
    throughput_hist_running rolling histogram of recent throughput
    reward_hist_running     rolling histogram of recent rewards
"""

import numpy as np
import gymnasium as gym

try:
    import wandb
except ImportError:
    wandb = None


class ThroughputLogger(gym.Wrapper):
    def __init__(self, env, enabled=False, running_window=4000):
        super().__init__(env)
        self.enabled = enabled
        self._running_window = running_window
        self._reset_ep_buffers()
        self._tp_running = []
        self._rew_running = []

    def _reset_ep_buffers(self):
        self._ep_rewards = []
        self._ep_throughputs = []
        self._ep_utilizations = []
        self._ep_movement = 0.0

    def _active(self):
        return self.enabled and wandb is not None and wandb.run is not None

    def _max_possible(self):
        """Sum of offered loads = theoretical max throughput. Guards missing attr."""
        try:
            return float(self.unwrapped.network.getMaxThroughput())
        except Exception:
            return 0.0

    def _action_norm(self, action):
        """Mean L2 norm across agents. Handles dict (PettingZoo) or array actions."""
        try:
            if isinstance(action, dict):
                arrs = [np.asarray(v, dtype=float) for v in action.values()]
            else:
                arrs = [np.asarray(action, dtype=float)]
            norms = [np.linalg.norm(a.ravel()) for a in arrs]
            return float(np.mean(norms)) if norms else 0.0
        except Exception:
            return 0.0

    def _roll(self, buf, val):
        buf.append(val)
        if len(buf) > self._running_window:
            buf.pop(0)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self._active():
            return obs, reward, terminated, truncated, info

        tp = float(self.unwrapped.network.getThroughput())
        r = float(reward)
        max_tp = self._max_possible()
        util = (tp / max_tp) if max_tp > 1e-9 else 0.0
        amove = self._action_norm(action)

        self._ep_rewards.append(r)
        self._ep_throughputs.append(tp)
        self._ep_utilizations.append(util)
        self._ep_movement += amove
        self._roll(self._tp_running, tp)
        self._roll(self._rew_running, r)

        wandb.log({
            "diag/throughput_raw": tp,
            "diag/throughput_mbps": tp / 1e6,
            "diag/throughput_utilization": util,
            "diag/reward": r,
            "diag/reward_abs": abs(r),
            "diag/action_norm_mean": amove,
        }, commit=False)

        if terminated or truncated:
            rr = np.array(self._ep_rewards)
            ra = np.abs(rr)
            tt = np.array(self._ep_throughputs)
            uu = np.array(self._ep_utilizations)
            med = float(np.median(ra)) if ra.size else 0.0
            mx = float(np.max(ra)) if ra.size else 0.0

            wandb.log({
                "diag/ep_len": int(rr.size),
                "diag/ep_outcome": 1 if terminated else 0,
                "diag/ep_reward_max": float(np.max(rr)) if rr.size else 0.0,
                "diag/ep_reward_min": float(np.min(rr)) if rr.size else 0.0,
                "diag/ep_reward_mean": float(np.mean(rr)) if rr.size else 0.0,
                "diag/ep_reward_median": float(np.median(rr)) if rr.size else 0.0,
                "diag/ep_reward_std": float(np.std(rr)) if rr.size else 0.0,
                "diag/ep_max_median_ratio": (mx / med) if med > 1e-9 else 0.0,
                "diag/ep_throughput_max": float(np.max(tt)) if tt.size else 0.0,
                "diag/ep_throughput_mean": float(np.mean(tt)) if tt.size else 0.0,
                "diag/ep_utilization_mean": float(np.mean(uu)) if uu.size else 0.0,
                "diag/ep_total_movement": float(self._ep_movement),
                "diag/ep_reward_hist": wandb.Histogram(rr.tolist()) if rr.size else None,
                "diag/ep_throughput_hist": wandb.Histogram(tt.tolist()) if tt.size else None,
                "diag/throughput_hist_running": wandb.Histogram(self._tp_running),
                "diag/reward_hist_running": wandb.Histogram(self._rew_running),
            }, commit=False)
            self._reset_ep_buffers()

        return obs, reward, terminated, truncated, info
