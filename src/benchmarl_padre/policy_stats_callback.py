"""PolicyStatsCallback — logs policy sigma/|loc| so the scale_lb floor
(benchmarl_padre/normalized_mappo.py) is verifiable from the training logs
instead of trusted blindly.

BenchMARL does not log the policy's standard deviation (sigma) or its mean
action magnitude (|loc|). Depending on entropy_coef, sigma can stay pinned at
its init value (~1.0) or collapse to ~0.002-0.009. This callback makes that
observable in real time, every collected batch, at negligible cost.

Kept dependency-light and at module top level (only stdlib + torch/
tensordict, no project imports) because Experiment.setup() pickles its
callback list into <run_folder>/config.pkl (benchmarl/experiment/
experiment.py:616) and unpickles it on --resume (omnet_checkpoint.py). A
callback that imports something heavier, or that isn't importable at this
exact module path, would make old checkpoints un-resumable.
"""
import warnings

from tensordict import TensorDictBase

from benchmarl.experiment.callback import Callback


class PolicyStatsCallback(Callback):
    """Logs policy/{group}/scale_mean, scale_min, abs_loc_mean per collected batch.

    Reads the (group, "scale") and (group, "loc") entries that the
    ProbabilisticActor's NormalParamExtractor writes into the rollout
    tensordict during collection (see benchmarl/algorithms/mappo.py's
    extractor_module, out_keys=[(group, "loc"), (group, "scale")]). These
    keys only exist for continuous action groups; discrete groups (plain
    Categorical/MaskedCategorical policies) have no such keys and are
    silently skipped -- this is a diagnostic, not a correctness gate (that
    job belongs to NormalizedMappo.get_policy_for_loss's RuntimeError).

    This is a diagnostic callback: it must NEVER be able to kill a
    multi-hour training run. Every batch's work is wrapped in a blanket
    try/except that logs a one-time warning on failure and otherwise does
    nothing -- degrade silently rather than crash the caller.
    """

    def __init__(self):
        super().__init__()
        self._warned = False

    def on_batch_collected(self, batch: TensorDictBase):
        try:
            self._log_policy_stats(batch)
        except Exception as exc:  # noqa: BLE001 -- diagnostic must never propagate
            if not self._warned:
                warnings.warn(
                    f"PolicyStatsCallback: failed to log policy stats "
                    f"({exc!r}). Disabling further warnings for this run; "
                    "training continues unaffected.",
                    stacklevel=2,
                )
                self._warned = True

    def _log_policy_stats(self, batch: TensorDictBase):
        if self.experiment is None:
            return  # not attached to a live experiment (e.g. unit tests)

        existing_keys = set(batch.keys(True, True))
        to_log = {}
        for group in self.experiment.group_map.keys():
            scale_key = (group, "scale")
            loc_key = (group, "loc")
            # Verify the keys actually exist before relying on them -- they
            # are absent for discrete-action groups, and might disappear
            # entirely if a future BenchMARL version changes what the actor
            # writes into the rollout tensordict.
            if scale_key not in existing_keys or loc_key not in existing_keys:
                continue

            scale = batch.get(scale_key).detach()
            loc = batch.get(loc_key).detach()
            to_log[f"policy/{group}/scale_mean"] = scale.mean().item()
            to_log[f"policy/{group}/scale_min"] = scale.min().item()
            to_log[f"policy/{group}/abs_loc_mean"] = loc.abs().mean().item()

        if to_log:
            self.experiment.logger.log(to_log, step=self.experiment.n_iters_performed)
