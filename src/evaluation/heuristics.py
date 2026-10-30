"""Heuristic baseline action classes for PADRE evaluation.

Two baselines implemented as callable classes with a reset(env, seed) protocol:

  GreedyHeuristic  — probe-greedy, faithful to MANET/Agents.py GreedyMaximizer.
                     Each drone independently probes 8 directions, returns to origin,
                     then commits to the best direction.

  SPSAHeuristic    — Simultaneous Perturbation Stochastic Approximation.
                     Each drone uses a 3-step cycle (probe+, probe-, commit) to
                     estimate and ascend the local condition gradient.

Both implement the action_fn(obs_dict, n_agents) -> action_dict interface used by
runner.py, plus a reset(env, seed) method that runner.py calls at episode start.

The `last_committed` attribute is updated each call so runner.py can log
fraction_committed_steps without coupling to heuristic internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Probe directions (verbatim from Agents.py AgentBase.poss_moves, ordered)
POSS_MOVES = [
    np.array([0.0,   1.0],  dtype=np.float32),  # up
    np.array([-0.5,  0.5],  dtype=np.float32),  # up-left
    np.array([-1.0,  0.0],  dtype=np.float32),  # left
    np.array([-0.5, -0.5],  dtype=np.float32),  # down-left
    np.array([0.0,  -1.0],  dtype=np.float32),  # down
    np.array([0.5,  -0.5],  dtype=np.float32),  # down-right
    np.array([1.0,   0.0],  dtype=np.float32),  # right
    np.array([0.5,   0.5],  dtype=np.float32),  # up-right
]
_N_DIRS = len(POSS_MOVES)  # 8
_STAY = np.zeros(2, dtype=np.float32)


def _make_strategy(condition_name: str, threshold=None):
    """Build a ConditionStrategy from a string name."""
    from MANET.ConditionStrategy import (
        SINRConditionStrategy,
        SignalConditionStrategy,
        InterferenceConditionStrategy,
        TrafficHostedConditionStrategy,
    )
    lut = {
        "sinr":         (SINRConditionStrategy,         {"threshold_sinr": threshold}    if threshold is not None else {}),
        "signal":       (SignalConditionStrategy,        {"threshold_signal": threshold}  if threshold is not None else {}),
        "interference": (InterferenceConditionStrategy,  {"threshold_interference": threshold} if threshold is not None else {}),
        "traffic":      (TrafficHostedConditionStrategy, {"traffic_threshold": threshold} if threshold is not None else {}),
    }
    if condition_name not in lut:
        raise ValueError(f"Unknown condition '{condition_name}'. Choose: {list(lut)}")
    cls, kwargs = lut[condition_name]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Greedy Heuristic
# ---------------------------------------------------------------------------

@dataclass
class _GreedyDroneState:
    phase: str = "IDLE"          # IDLE | PROBE | RETURN
    probe_idx: int = 0           # next probe direction index to visit (0-7 in PROBE)
    last_rel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    samples: dict = field(default_factory=dict)
    init_pos: Optional[np.ndarray] = None  # absolute position at probe start
    return_action: Optional[np.ndarray] = None  # pre-computed in PROBE(7)→RETURN

    def reset(self):
        self.phase = "IDLE"
        self.probe_idx = 0
        self.last_rel = np.zeros(2, dtype=np.float32)
        self.samples = {}
        self.init_pos = None
        self.return_action = None


class GreedyHeuristic:
    """Probe-greedy heuristic — faithful reimplementation of GreedyMaximizer.

    Protocol per drone (independent per drone):
      IDLE:   check condition. If viable → stay. If not → begin probing.
      PROBE:  visit 8 directions in turn, reading condition AFTER each move.
      RETURN: move back toward origin, read condition there, commit to best dir.

    The return-to-origin action is `init_pos - node.pos` (identical to
    GreedyMaximizer's `back_to_init_pos`), which lands approx 1 env unit
    short of the exact origin due to step_size scaling — faithful, not fixed.
    """

    def __init__(self, condition_name: str = "sinr", threshold=None):
        self.condition_name = condition_name
        self.threshold = threshold
        self._strategy = None
        self._states: list[_GreedyDroneState] = []
        self._nodes: list = []
        self.last_committed: bool = False  # runner reads this for fraction_committed metric

    def reset(self, env, seed=None):
        """Called by runner at the start of each episode with the fresh env."""
        base = env._wrapper.unwrapped_env
        self._nodes = list(base.network.nodes)
        n = len(self._nodes)
        self._strategy = _make_strategy(self.condition_name, self.threshold)
        if len(self._states) != n:
            self._states = [_GreedyDroneState() for _ in range(n)]
        else:
            for s in self._states:
                s.reset()

    def __call__(self, obs_dict: dict, n_agents: int) -> dict:
        committed_any = False
        actions = {}
        for i in range(n_agents):
            a, committed = self._step_drone(i)
            actions[f"drone_{i}"] = a
            committed_any = committed_any or committed
        self.last_committed = committed_any
        return actions

    # ------------------------------------------------------------------

    def _step_drone(self, i: int) -> tuple[np.ndarray, bool]:
        """Advance drone i's state machine by one step. Returns (action, committed)."""
        state = self._states[i]
        node = self._nodes[i]

        if state.phase == "IDLE":
            return self._idle(state, node)
        elif state.phase == "PROBE":
            return self._probe(state, node)
        else:  # RETURN
            return self._return(state, node)

    def _idle(self, state: _GreedyDroneState, node) -> tuple:
        if not self._strategy.is_unviable(node):
            return _STAY.copy(), True  # viable → stay (committed decision)

        # Condition is unviable — begin probing
        state.init_pos = node.pos.copy()
        state.last_rel = np.zeros(2, dtype=np.float32)
        state.samples = {}
        state.probe_idx = 0
        state.phase = "PROBE"

        desired = POSS_MOVES[0]
        action = desired - state.last_rel
        state.last_rel = desired.copy()
        return action.copy(), False  # probe move — not committed

    def _probe(self, state: _GreedyDroneState, node) -> tuple:
        # We just moved to POSS_MOVES[probe_idx] relative position.
        # Read condition here (this is the post-env-step state).
        current_desired = POSS_MOVES[state.probe_idx]
        state.samples[tuple(current_desired)] = self._strategy.sample(node)

        state.probe_idx += 1

        if state.probe_idx < _N_DIRS:
            # Advance to next probe
            next_desired = POSS_MOVES[state.probe_idx]
            action = next_desired - state.last_rel
            state.last_rel = next_desired.copy()
            return action.copy(), False

        # All 8 probes done — compute return action and transition to RETURN.
        # `init_pos - node.pos` is in absolute position units; the env scales
        # by step_size=2, so this slightly undershoots the true origin (faithful
        # to GreedyMaximizer's `back_to_init_pos = init_pos - node.pos`).
        state.return_action = state.init_pos - node.pos
        state.phase = "RETURN"
        return state.return_action.copy(), False

    def _return(self, state: _GreedyDroneState, node) -> tuple:
        # We have returned (approximately) to the origin. Read condition.
        state.samples[(0.0, 0.0)] = self._strategy.sample(node)

        # Commit to the best probe direction (including staying at origin).
        best_key = max(state.samples, key=state.samples.get)
        best_dir = np.array(best_key, dtype=np.float32)

        state.phase = "IDLE"

        if np.allclose(best_dir, _STAY):
            return _STAY.copy(), True
        return best_dir, True  # commit move


# ---------------------------------------------------------------------------
# SPSA Heuristic
# ---------------------------------------------------------------------------

@dataclass
class _SPSADroneState:
    phase: str = "GENERATE"  # GENERATE | PROBE_PLUS | PROBE_MINUS
    delta: Optional[np.ndarray] = None   # Rademacher perturbation vector
    y_plus: float = 0.0

    def reset(self):
        self.phase = "GENERATE"
        self.delta = None
        self.y_plus = 0.0


class SPSAHeuristic:
    """SPSA (Simultaneous Perturbation Stochastic Approximation) heuristic.

    3-step cycle per drone:
      GENERATE:    draw Δ ~ Rademacher(2); emit +c*Δ (move to +probe).
      PROBE_PLUS:  read y+;  emit -2c*Δ (move to -probe).
      PROBE_MINUS: read y-;  compute ĝ = (y+ - y-) / (2c*Δ);
                             emit clip(a*ĝ, -1,1) + c*Δ (return to base + gradient step).

    Net movement per cycle (in env units): step_size * clip(a*ĝ, -1,1).
    The probe displacements +c*Δ and -c*Δ cancel exactly.

    Parameters:
      c   perturbation size in action units (default 1.0 → ±2 env units per axis)
      a   gradient ascent step gain (default 1.0)
    """

    def __init__(
        self,
        condition_name: str = "sinr",
        c: float = 1.0,
        a: float = 1.0,
        threshold=None,
    ):
        self.condition_name = condition_name
        self.c = c
        self.a = a
        self.threshold = threshold
        self._strategy = None
        self._states: list[_SPSADroneState] = []
        self._nodes: list = []
        self._rng: Optional[np.random.Generator] = None
        self.last_committed: bool = False

    def reset(self, env, seed=None):
        """Called by runner at the start of each episode."""
        base = env._wrapper.unwrapped_env
        self._nodes = list(base.network.nodes)
        n = len(self._nodes)
        self._strategy = _make_strategy(self.condition_name, self.threshold)
        # Seed RNG per episode for reproducibility
        self._rng = np.random.default_rng(seed)
        if len(self._states) != n:
            self._states = [_SPSADroneState() for _ in range(n)]
        else:
            for s in self._states:
                s.reset()

    def __call__(self, obs_dict: dict, n_agents: int) -> dict:
        committed_any = False
        actions = {}
        for i in range(n_agents):
            a, committed = self._step_drone(i)
            actions[f"drone_{i}"] = a
            committed_any = committed_any or committed
        self.last_committed = committed_any
        return actions

    # ------------------------------------------------------------------

    def _step_drone(self, i: int) -> tuple[np.ndarray, bool]:
        state = self._states[i]
        node = self._nodes[i]

        if state.phase == "GENERATE":
            return self._generate(state)
        elif state.phase == "PROBE_PLUS":
            return self._probe_plus(state, node)
        else:  # PROBE_MINUS
            return self._probe_minus(state, node)

    def _generate(self, state: _SPSADroneState) -> tuple:
        # Draw Rademacher vector: each component ±1 with equal probability
        state.delta = self._rng.choice([-1.0, 1.0], size=2).astype(np.float32)
        action = (self.c * state.delta).astype(np.float32)
        state.phase = "PROBE_PLUS"
        return action, False  # moving to + probe, not committed

    def _probe_plus(self, state: _SPSADroneState, node) -> tuple:
        # We just moved to +c*Δ from last base. Read y+.
        state.y_plus = self._strategy.sample(node)
        # Move to -c*Δ from base (i.e., -2c*Δ from current position)
        action = (-2.0 * self.c * state.delta).astype(np.float32)
        state.phase = "PROBE_MINUS"
        return action, False  # moving to - probe, not committed

    def _probe_minus(self, state: _SPSADroneState, node) -> tuple:
        # We just moved to -c*Δ from last base. Read y-.
        y_minus = self._strategy.sample(node)

        # Gradient estimate: ĝ_i = (y+ - y-) / (2*c*Δ_i)
        g_hat = (state.y_plus - y_minus) / (2.0 * self.c * state.delta)

        # Gradient ascent step (clipped to action range [-1,1])
        ascent = np.clip(self.a * g_hat, -1.0, 1.0).astype(np.float32)

        # Commit action: return from -c*Δ to base (+c*Δ) then apply ascent step.
        # Net: c*Δ + ascent  (the probe displacement cancels over the full 3-step cycle)
        action = (self.c * state.delta + ascent).astype(np.float32)
        state.phase = "GENERATE"
        return action, True  # commit step
