"""CLI entry point for PADRE policy evaluation.

Usage (from src/):
    python -m evaluation.evaluate <checkpoint.pt>                       # visual, 3 eps
    python -m evaluation.evaluate <checkpoint.pt> --no-render -n 20    # headless batch
    python -m evaluation.evaluate <checkpoint.pt> --policy random       # random baseline
    python -m evaluation.evaluate <checkpoint.pt> --policy greedy       # greedy heuristic
    python -m evaluation.evaluate <checkpoint.pt> --policy greedy --condition sinr
    python -m evaluation.evaluate <checkpoint.pt> --policy spsa --spsa-c 1.0 --spsa-a 1.0
    python -m evaluation.evaluate <checkpoint.pt> --random              # legacy alias for --policy random
    python -m evaluation.evaluate --policy greedy --condition sinr      # heuristics need NO checkpoint

Results written to src/evaluation/results/<checkpoint>_<policy[_condition]>_<timestamp>/
  stats.csv          aggregate metric | mean | std (thesis table)
  per_episode.csv    one row per seeded episode, all metrics
  config_used.yaml   snapshot of env config (provenance)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Ensure src/ is on path when run as `python -m evaluation.evaluate`
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PADRE/BenchMARL policy.")
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="Path to a BenchMARL .pt checkpoint. Required only for --policy trained.")
    p.add_argument(
        "--render", dest="render", action="store_true", default=False,
        help="Open the PADRE plot GUI for the first episode (default ON).",
    )
    p.add_argument(
        "--no-render", dest="render", action="store_false",
        help="Run headless (no GUI).",
    )
    p.add_argument(
        "-n", "--episodes", type=int, default=None,
        help="Number of seeded episodes. Default: 3 with render, 20 without.",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit seed list. Defaults to range(n).",
    )
    p.add_argument(
        "--seed-start", type=int, default=0, dest="seed_start",
        help="First seed for the default range(n) (default 0); episode i "
             "uses seed seed_start + i. Exists to shard evaluation across "
             "parallel tasks -- each task takes a disjoint seed slice, "
             "since every row carries its own seed. SEED DISCIPLINE: 0-99 "
             "is the evaluation band; 100-199 is tuning-only. Straying "
             "past 99 is refused rather than silently allowed. Ignored "
             "when --seeds is given explicitly.",
    )
    p.add_argument(
        "--policy", choices=["trained", "random", "greedy", "spsa"], default="trained",
        help="Which policy to evaluate (default: trained).",
    )
    p.add_argument(
        "--random", action="store_true", default=False,
        help="Legacy alias for --policy random.",
    )
    p.add_argument(
        "--condition", choices=["sinr", "signal", "interference", "traffic"],
        default="sinr",
        help="Condition strategy for heuristic baselines (default: sinr).",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Condition-strategy threshold (e.g. SINR dB for --condition sinr). "
             "Default: the strategy's own default (10 dB for sinr).",
    )
    p.add_argument(
        "--spsa-c", type=float, default=1.0, dest="spsa_c",
        help="SPSA perturbation size in action units (default 1.0).",
    )
    p.add_argument(
        "--spsa-a", type=float, default=1.0, dest="spsa_a",
        help="SPSA gradient ascent gain (default 1.0).",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Compute device (cuda/cpu). Auto-detected if not specified.",
    )
    p.add_argument(
        "--numb-of-nodes", type=int, default=None, dest="numb_of_nodes",
        help="Override numbOfNodes from task config (scalar only). "
             "Use to evaluate a checkpoint at an N different from training "
             "(e.g. FFN trained at 4, eval at 5 — expected to fail with "
             "tensor-shape mismatch; that IS the result).",
    )
    p.add_argument(
        "--env-class", type=str, default=None, choices=["static", "dynamic"],
        dest="env_class",
        help="Override env class (static|dynamic). Default: from task config.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    import os as _os
    if getattr(args, 'render', False) and not _os.environ.get('DISPLAY'):
        raise SystemExit(
            'FATAL: --render requested but $DISPLAY is unset (headless). '
            'The GUI thread dies and the job burns its wall doing nothing. '
            'Use --no-render for batch/sbatch runs. [bug 25]')

    # --random is a legacy alias for --policy random
    policy_mode = "random" if args.random else args.policy

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")


    n_episodes = args.episodes or (3 if args.render else 20)

    if args.seeds is not None:
        seeds = args.seeds
    else:
        if args.seed_start < 0:
            sys.exit(f"error: --seed-start {args.seed_start} is negative.")
        _last_seed = args.seed_start + n_episodes - 1
        if _last_seed > 99:
            sys.exit(
                f"error: seeds {args.seed_start}..{_last_seed} reach past 99 "
                "into the tuning band (100-199). Reported numbers come from "
                "seeds 0-99 only; mixing the bands invalidates the cell. "
                "Reduce -n or --seed-start."
            )
        seeds = list(range(args.seed_start, args.seed_start + n_episodes))

    if policy_mode == "trained" and args.checkpoint is None:
        sys.exit("error: --policy trained requires a checkpoint argument")

    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else None

    # ------------------------------------------------------------------
    # Load / construct policy
    # ------------------------------------------------------------------
    encoder_meta = None  # only set for policy_mode == "trained" (see below)
    if policy_mode == "random":
        from evaluation.policies import make_random_action_fn
        from MANET.build import build_padre_env
        from MANET.Parallel_enviroments import PADREParallelEnv

        _probe = PADREParallelEnv(build_env_fn=lambda: build_padre_env(numbOfUsers=int(os.environ.get("PADRE_EVAL_USERS", 15))))
        n_agents = len(_probe.possible_agents)
        try:
            _probe.close()
        except Exception:
            pass

        action_fn = make_random_action_fn(n_agents)
        task_config = _get_default_task_config()
        policy_label = "random"

    elif policy_mode == "greedy":
        from evaluation.policies import make_greedy_action_fn
        action_fn = make_greedy_action_fn(condition_name=args.condition,
                                          threshold=args.threshold)
        task_config = _get_default_task_config()
        policy_label = (f"greedy_{args.condition}_t{args.threshold}"
                        if args.threshold is not None else f"greedy_{args.condition}")

    elif policy_mode == "spsa":
        from evaluation.policies import make_spsa_action_fn
        action_fn = make_spsa_action_fn(
            condition_name=args.condition,
            c=args.spsa_c,
            a=args.spsa_a,
        )
        task_config = _get_default_task_config()
        policy_label = f"spsa_{args.condition}"

    else:  # trained
        print(f"Loading checkpoint: {ckpt_path}")
        from evaluation.policies import load_trained_policy, inspect_checkpoint_encoder
        action_fn, task_config = load_trained_policy(str(ckpt_path), device)
        policy_label = "trained"
        print(f"Policy loaded. Task config: {task_config}")

        # config.pkl reload can "succeed" while persisting almost nothing --
        # OMNeT-launched runs (benchmarl_padre.omnet_common.OMNeTClass) only
        # ever write {'ep_length': ...} to task.config, so numbOfNodes/
        # numbOfUsers/obs_wrapper/mask/etc are silently absent BY DESIGN
        # (OMNeT's obs shape comes from omnet_env.py module constants, not
        # task config -- see runner._build_env). That's only a problem for
        # PADRE checkpoints, where the same absence means config.pkl didn't
        # record the run and _build_env would silently fall back to
        # build_padre_env's cooked-4-node defaults -- an eval that runs to
        # completion and reports numbers for the WRONG env, not a crash.
        # Fail loud instead of guessing, but only for the family that
        # actually needs these keys.
        if task_config.get("_env_family") != "omnet":
            _REQUIRED_TASK_KEYS = {"numbOfNodes", "numbOfUsers"}
            _missing = _REQUIRED_TASK_KEYS - task_config.keys()
            if _missing:
                sys.exit(
                    f"error: this checkpoint's config.pkl did not persist "
                    f"{sorted(_missing)} (task_config={task_config}). The run's "
                    f"config wasn't recorded (common for OMNeT-launched training), "
                    f"so reload_from_file can't recover the real env params -- "
                    f"building an env from this task_config would silently use "
                    f"the wrong obs_wrapper/numbOfNodes rather than error. Evaluate "
                    f"from a standalone .pt via the hydra-compose fallback instead "
                    f"(copy the checkpoint out from beside its config.pkl), or run "
                    f"this eval on-cluster where the original run dir is intact."
                )
        encoder_meta = inspect_checkpoint_encoder(str(ckpt_path))

    # --numb-of-nodes / --env-class only mean anything to build_padre_env
    # (runner._build_env's PADRE branch). OMNeT's drone count and env class
    # are omnet_env.py module constants (MAX_N / the fixed sim topology),
    # never read from task_config -- silently setting these keys on an
    # OMNeT task_config would be a no-op that looks like it did something.
    if task_config.get("_env_family") == "omnet":
        if args.numb_of_nodes is not None:
            sys.exit(
                "error: --numb-of-nodes has no effect on an OMNeT checkpoint "
                "-- drone count is the module constant MAX_N in "
                "omnet_module/omnet_env.py (override via $OMNET_NUM_DRONES "
                "before the run, not this flag)."
            )
        if args.env_class is not None:
            sys.exit(
                "error: --env-class has no effect on an OMNeT checkpoint -- "
                "there is no static/dynamic split in omnet_env.py."
            )
    else:
        # CLI override for numbOfNodes — applies to all non-OMNeT policy modes.
        if args.numb_of_nodes is not None:
            old_n = task_config.get("numbOfNodes")
            task_config["numbOfNodes"] = args.numb_of_nodes
            print(f"[override] numbOfNodes: {old_n} -> {args.numb_of_nodes}")

        if args.env_class is not None:
            old_e = task_config.get("env_class", "static")
            task_config["env_class"] = args.env_class
            print(f"[override] env_class: {old_e} -> {args.env_class}")

    # ------------------------------------------------------------------
    # Run episodes
    # ------------------------------------------------------------------
    from evaluation.runner import run_episodes

    print(f"\nRunning {len(seeds)} seeded episode(s)  "
          f"[policy={policy_label}, render={args.render}, device={device}]\n")

    per_episode = run_episodes(
        action_fn=action_fn,
        task_config=task_config,
        seeds=seeds,
        render=args.render,
        encoder_meta=encoder_meta,
    )

    # ------------------------------------------------------------------
    # Aggregate and display
    # ------------------------------------------------------------------
    from evaluation.metrics import aggregate

    stats = aggregate(per_episode)
    _print_stats_table(stats, n_episodes=len(seeds))

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_dir = _make_results_dir(ckpt_path, policy_label)
    _write_per_episode_csv(out_dir / "per_episode.csv", per_episode)
    _write_stats_csv(out_dir / "stats.csv", stats)
    _write_config_yaml(out_dir / "config_used.yaml", task_config, args, policy_label)
    print(f"\nResults written to: {out_dir}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _make_results_dir(ckpt_path, policy_label: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = ckpt_path.stem if ckpt_path is not None else "nockpt"
    name = f"{stem}_{policy_label}_{ts}"
    results_root = Path(__file__).parent / "results"
    out_dir = results_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_per_episode_csv(path: Path, per_episode: list[dict]) -> None:
    if not per_episode:
        return
    fieldnames = ["episode"] + [k for k in per_episode[0] if k != "episode"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(per_episode):
            writer.writerow({"episode": i, **row})


def _write_stats_csv(path: Path, stats: dict) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for metric, (mean, std) in stats.items():
            writer.writerow([metric, f"{mean:.6f}", f"{std:.6f}"])


def _write_config_yaml(path: Path, task_config: dict, args, policy_label: str) -> None:
    record = {
        "task_config": task_config,
        "eval_args": {
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "policy": policy_label,
            "render": args.render,
            "episodes": args.episodes,
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "condition": getattr(args, "condition", None),
            "spsa_c": getattr(args, "spsa_c", None),
            "spsa_a": getattr(args, "spsa_a", None),
        },
    }
    with open(path, "w") as f:
        yaml.dump(record, f, default_flow_style=False)


def _print_stats_table(stats: dict, n_episodes: int) -> None:
    print(f"\n{'─'*56}")
    print(f"  Aggregate statistics  ({n_episodes} seeded episodes)")
    print(f"{'─'*56}")
    print(f"  {'metric':<30}  {'mean':>10}  {'std':>8}")
    print(f"{'─'*56}")
    for metric, (mean, std) in stats.items():
        if metric == "seed":
            continue
        print(f"  {metric:<30}  {mean:10.4f}  {std:8.4f}")
    print(f"{'─'*56}\n")


def _get_default_task_config() -> dict:
    """Return the canonical task config from src/conf/task/padre/static.yaml."""
    conf_path = Path(__file__).parent.parent / "conf" / "task" / "padre" / "static.yaml"
    with open(conf_path) as f:
        raw = yaml.safe_load(f)
    # Strip Hydra defaults block if present
    raw.pop("defaults", None)
    return raw


if __name__ == "__main__":
    main()
