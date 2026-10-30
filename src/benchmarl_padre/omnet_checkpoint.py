"""Resume-from-checkpoint support for OMNeT training (src/run.py only).

BenchMARL's own Experiment.state_dict()/load_state_dict() checkpoints the
loss modules (policy + critic weights), replay buffers, and the collector,
plus the total_frames/n_iters_performed counters -- but NOT the optimizers.
Adam's exp_avg/exp_avg_sq moment estimates are silently dropped on any
restore_file-based reload. A resumed run trains with cold Adam state unless
something explicitly saves/restores it; this module is that something.

OptimizerStateCallback plugs into BenchMARL's Callback hooks (on_state_dict /
on_load_state_dict) to add optimizer state into every checkpoint and restore
it on reload. It is architecture-agnostic (keyed by group/loss_name, not by
model type), so it applies the same way to ST, FFN, and DeepSets.

resume_experiment() rebuilds an Experiment from a checkpoint file the same
way benchmarl.experiment.Experiment.reload_from_file does -- reading
config.pkl from two levels above the checkpoint so the task/model/algorithm
config is the exact pickled object graph from the original run, not a
freshly hydra-parsed one. This is deliberate: BenchMARL's restore_file path
raises tensordict AttributeError: __batch_size for the ST model when the
experiment is instead rebuilt from a hydra-composed config (as e.g.
`+experiment.restore_file=...` would do). Going through config.pkl -- the
same path src/evaluation/policies.py already uses for ST eval -- avoids it.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from benchmarl.experiment import Callback, Experiment


class OptimizerStateCallback(Callback):
    """Adds/restores Adam optimizer state alongside BenchMARL's own checkpoint."""

    _KEY = "omnet_resume_optimizers"

    def on_state_dict(self, state_dict: dict) -> None:
        # This runs mid-run at EVERY checkpoint write. experiment.optimizers is
        # not a documented BenchMARL API, so a version change could make this
        # raise inside the checkpoint path and kill a 40h job. Optimizer state
        # is a nice-to-have (cold Adam costs re-warmup, not correctness); a
        # dead run is not. Fail loud, keep training.
        try:
            state_dict[self._KEY] = {
                group: {name: opt.state_dict() for name, opt in opts.items()}
                for group, opts in self.experiment.optimizers.items()
            }
        except Exception as exc:  # noqa: BLE001
            print(
                f"\n[OptimizerStateCallback] WARNING: could not capture optimizer "
                f"state ({type(exc).__name__}: {exc}). Checkpoint is still valid "
                f"for weights and frame counters; a resume from it will start Adam "
                f"cold. Training continues.\n",
                flush=True,
            )

    def on_load_state_dict(self, state_dict: dict) -> None:
        saved = state_dict.get(self._KEY)
        if saved is None:
            # Checkpoint predates this callback (e.g. the first resume of a
            # run started before this file existed). Optimizer state was
            # never saved, so there is nothing to restore -- Adam starts
            # cold this one time. Every checkpoint written from here on
            # (this callback is now attached) carries it, so the NEXT
            # resume off one of those restores correctly.
            print(
                f"\n[OptimizerStateCallback] no '{self._KEY}' in checkpoint -- "
                "optimizer state NOT restored (checkpoint predates this callback). "
                "Adam moments start cold; policy/critic weights and frame "
                "counters are unaffected.\n"
            )
            return
        for group, opts in self.experiment.optimizers.items():
            for name, opt in opts.items():
                if group not in saved or name not in saved[group]:
                    raise KeyError(
                        f"checkpoint has no optimizer state for group={group!r} "
                        f"name={name!r}; checkpoint groups="
                        f"{ {g: list(v) for g, v in saved.items()} }. This means "
                        f"the checkpoint was written by a differently-shaped "
                        f"experiment (different arch or N?)."
                    )
                opt.load_state_dict(saved[group][name])
        print(
            f"\n[OptimizerStateCallback] restored optimizer state for groups "
            f"{list(self.experiment.optimizers.keys())} from '{self._KEY}'.\n"
        )
        # load_state_dict restores the SAVED param_group lr too, undoing any
        # resume-time override -- re-pin it (see _force_lr).
        force = getattr(self, "force_lr", None)
        if force is not None:
            _force_lr(self.experiment, force, "after optimizer state restore")


def _force_lr(experiment: Experiment, lr: float, where: str) -> None:
    """Pin every optimizer param_group to lr and print what is now in effect.

    Resume rebuilds the Experiment from config.pkl, so without this the
    optimizer is rebuilt at config.pkl's lr (e.g. 5e-5) even when the cell
    file sets experiment.lr=1e-3. OptimizerStateCallback.load_state_dict ALSO
    restores the saved param_group lr, so this must run after that too.
    """
    seen = []
    for group, opts in experiment.optimizers.items():
        for name, opt in opts.items():
            for pg in opt.param_groups:
                pg["lr"] = lr
            seen.append((group, name, [pg["lr"] for pg in opt.param_groups]))
    print(f"\n[resume_experiment] lr FORCED to {lr} ({where}); param_groups now: {seen}\n",
          flush=True)


def resume_experiment(
    restore_file: str,
    max_n_frames: Optional[int] = None,
    lr: Optional[float] = None,
) -> Experiment:
    """Reconstruct an Experiment from a checkpoint, resuming training in place.

    Mirrors benchmarl.experiment.Experiment.reload_from_file's config.pkl
    logic exactly, with one addition: an OptimizerStateCallback is appended
    to the callback list (if not already present) so optimizer state is
    saved/restored going forward, which reload_from_file does not do.

    Args:
        restore_file: path to checkpoints/checkpoint_<N>.pt. config.pkl is
            expected at restore_file/../../config.pkl (the standard BenchMARL
            experiment folder layout).
        max_n_frames: if given, OVERWRITES the total frame budget (not
            additive -- BenchMARL's max_n_frames is always a total across
            resumes). Leave None to keep training toward the original run's
            total unchanged.
        lr: if given, OVERRIDES config.pkl's learning rate for the resumed
            run (config + every optimizer param_group). Without it a resume
            trains at the pickled lr regardless of experiment.lr.
    """
    ckpt = Path(restore_file).resolve()
    config_file = ckpt.parent.parent / "config.pkl"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No config.pkl at {config_file} -- restore_file must be "
            f"<experiment_folder>/checkpoints/checkpoint_<N>.pt."
        )

    with open(config_file, "rb") as f:
        task = pickle.load(f)
        task_config = pickle.load(f)
        algorithm_config = pickle.load(f)
        model_config = pickle.load(f)
        seed = pickle.load(f)
        experiment_config = pickle.load(f)
        critic_model_config = pickle.load(f)
        callbacks = list(pickle.load(f))

    # The eight loads above are POSITIONAL and mirror BenchMARL's own
    # reload_from_file. That makes this a second implementation of a format
    # defined elsewhere. If BenchMARL reorders or adds
    # a field, every variable would bind to the wrong object and the
    # experiment looks plausible. Assert the shape instead of trusting it.
    if not isinstance(seed, int):
        raise TypeError(
            f"config.pkl field 5 (seed) unpickled as {type(seed).__name__}, "
            f"expected int -- BenchMARL's config.pkl layout has changed and "
            f"resume_experiment() must be re-diffed against "
            f"Experiment.reload_from_file."
        )
    for _name, _obj in (("experiment_config", experiment_config),
                        ("algorithm_config", algorithm_config),
                        ("model_config", model_config)):
        if _obj is None or isinstance(_obj, (int, str, bytes)):
            raise TypeError(
                f"config.pkl field {_name} unpickled as "
                f"{type(_obj).__name__} -- layout changed, re-diff "
                f"resume_experiment() against Experiment.reload_from_file."
            )
    if not hasattr(experiment_config, "max_n_frames"):
        raise TypeError(
            "config.pkl experiment_config has no max_n_frames -- layout "
            "changed, re-diff resume_experiment()."
        )

    task.config = task_config
    experiment_config.restore_file = str(ckpt)
    if max_n_frames is not None:
        print(
            f"[resume_experiment] overriding max_n_frames: "
            f"{experiment_config.max_n_frames} -> {max_n_frames} (total, not additive)"
        )
        experiment_config.max_n_frames = max_n_frames
    if lr is not None:
        print(
            f"[resume_experiment] overriding lr: "
            f"{getattr(experiment_config, 'lr', None)} -> {lr}"
        )
        experiment_config.lr = lr

    if not any(isinstance(cb, OptimizerStateCallback) for cb in callbacks):
        callbacks.append(OptimizerStateCallback())
    if lr is not None:
        for cb in callbacks:
            if isinstance(cb, OptimizerStateCallback):
                cb.force_lr = lr

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=seed,
        config=experiment_config,
        callbacks=callbacks,
        critic_model_config=critic_model_config,
    )
    print(
        f"\nResumed experiment {experiment.name} from {ckpt}: "
        f"total_frames={experiment.total_frames}, "
        f"n_iters_performed={experiment.n_iters_performed}, "
        f"max_n_frames={experiment.config.max_n_frames}.\n"
    )
    if lr is not None:
        _force_lr(experiment, lr, "after Experiment build")
    return experiment
