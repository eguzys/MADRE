#!/usr/bin/env python3
"""
gen_stage2_job.py — fills stage2_train.sbatch's __TOKEN__ placeholders from a
Tier-B survivor JSON (sweep_config.py's --json output, or one entry of
harvest_configs.py's ranked list) and writes a ready-to-submit sbatch file.

The generated file is SELF-CONTAINED: the config's own median_P0v is baked
in as the pass/fail baseline (see check_stage2_baseline.py), so a later
reader never has to re-derive or re-look-up which config a given baseline
number belonged to.

Usage:
    python3 gen_stage2_job.py --config cfg_0001.json --arch st \\
        --max-frames 750000 --out stage2_st_n4_L1000.sbatch
    sbatch stage2_st_n4_L1000.sbatch   # on the cluster, after checking the
                                       # header comment's REQUIRED FILLS
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_knobs import Config, build_omnet_extra_args  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="Tier-B JSON (sweep_config.py --json output) for the "
                         "shortlisted config to train.")
    ap.add_argument("--arch", required=True, choices=["mlp", "st", "deepsets", "ffn"])
    ap.add_argument("--max-frames", type=int, default=750_000,
                    help="Stage-2 short-run budget (task brief: 500k-1M, not 3M).")
    ap.add_argument("--label", type=str, default="",
                    help="free-text tag for the save folder / job name; "
                         "defaults to N<n>_L<L>_mc<mc>.")
    ap.add_argument("--encoder-mode", choices=["learned", "frozen"], default="learned",
                    help="User-set encoder mode, exported as $OMNET_ENCODER_MODE. "
                         "'learned' (default) is the OMNeT design decision: the "
                         "env emits raw per-user tokens and the architecture pools "
                         "them with its own encoder, trained from the RL reward. "
                         "'frozen' is the round-1 path (env-side 8-dim "
                         "reconstruction autoencoder, obs_layout=padre). The "
                         "template exports the variable explicitly.")
    ap.add_argument("--template", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "stage2_train.sbatch"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg_json = json.load(f)

    required = ["N", "L", "max_clusters"]
    missing = [k for k in required if k not in cfg_json]
    if missing:
        raise SystemExit(f"--config {a.config} is missing required key(s) {missing} "
                         f"-- is this a sweep_config.py JSON (has 'N','L','max_clusters')?")

    cfg = Config(
        n_drones=int(cfg_json["N"]), L=float(cfg_json["L"]),
        max_clusters=int(cfg_json["max_clusters"]),
        p_cross=float(cfg_json.get("p_cross", 0.5)),
        user_spread_eff=float(cfg_json.get("user_spread_eff", 25.5)),
        step_m=float(cfg_json.get("step_m", 20.0)),
        drone_tx_mW=float(cfg_json.get("drone_tx_mW", 200.0)),
        user_tx_mW=float(cfg_json.get("user_tx_mW", 10.0)),
        ep_steps=int(cfg_json.get("ep_steps", 1000)),
    )
    extra_args = build_omnet_extra_args(cfg)

    p0v = cfg_json.get("median_P0v")
    if p0v is None:
        raise SystemExit(f"--config {a.config} has no 'median_P0v' -- this must be a "
                         f"sweep_config.py Tier-B JSON (P0v is what Tier C's success "
                         f"check compares against; a Tier-A screen_configs.py JSON "
                         f"has no measured P0v and cannot generate a Stage-2 job).")

    label = a.label or f"N{cfg.n_drones}_L{int(cfg.L)}_mc{cfg.max_clusters}"

    with open(a.template) as f:
        template = f.read()

    subs = {
        "__ARCH__": a.arch,
        "__NUM_DRONES__": str(cfg.n_drones),
        "__LABEL__": label,
        "__EXTRA_ARGS__": extra_args,
        "__DRONE_TX_MW__": str(cfg.drone_tx_mW),
        "__USER_TX_MW__": str(cfg.user_tx_mW),
        "__STEP_M__": str(cfg.step_m),
        "__MAX_FRAMES__": str(a.max_frames),
        "__P0V_BASELINE__": str(p0v),
        "__ENCODER_MODE__": a.encoder_mode,
    }
    out_text = template
    for tok, val in subs.items():
        out_text = out_text.replace(tok, val)

    remaining = [tok for tok in subs if tok in out_text]
    if remaining:
        raise SystemExit(f"internal error: tokens not fully substituted: {remaining}")

    with open(a.out, "w") as f:
        f.write(out_text)
    os.chmod(a.out, 0o755)

    print(f"wrote {a.out}")
    print(f"  arch={a.arch}  N={cfg.n_drones}  L={cfg.L:.0f}  mc={cfg.max_clusters}  "
         f"label={label}")
    print(f"  P0v baseline baked in: {p0v}")
    print(f"  max_frames: {a.max_frames}")
    print(f"  REMINDER: this is a template fill -- check stage2_train.sbatch's header "
         f"comment for the cluster-path REQUIRED FILLS (SCRATCH, conda env, sif path) "
         f"before submitting.")


if __name__ == "__main__":
    main()
