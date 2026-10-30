#!/usr/bin/env python3
"""
per_feature_recon.py

Descriptive (NOT a verdict): for each of the 6 user-token columns, report
  - the natural spread (std) of the real values, and
  - the encoder's reconstruction error (RMS),
both converted to that column's PHYSICAL UNITS, so the error is
interpretable ("column X naturally swings by A units, encoder blurs it by
B units"). No threshold, no pass/fail. Just characterization.

Reads the same frozen encoder + data as noise_floor.py.

Column physical units / normalization spans (from source):
  0 angle        : span depends on ANGLE_MINMAX (set below; check source)
  1 power        : signalMinMaxTuple (-100,-20) dBm  -> 80 dB span
  2 SNR          : snrMinMaxTuple    (-10, 40) dB    -> 50 dB span
  3 noise-ctx    : interferenceMinMaxTuple (-100,-20) -> 80 dB span
  4 thru-ingress : throughputUserMinMaxTuple (30,68) in 10log10(bits) -> 38 span
  5 thru-egress  : same as 4

Normalized token space is [-1,1] (width 2). A normalized quantity q maps to
physical:  phys = q/2 * span   (for a delta/std; offsets cancel for spread+error)
"""

import argparse, os, json, sys
import numpy as np
import torch

SWEEP_SCRIPT_DIR = os.environ.get("SWEEP_SCRIPT_DIR", "/cluster/scratch/eguzys")
sys.path.insert(0, SWEEP_SCRIPT_DIR)
from compare_encoders_fixed import DeepSetsAE

NPZ_X_KEY = os.environ.get("NPZ_X_KEY", "tokens")
NPZ_M_KEY = os.environ.get("NPZ_M_KEY", "masks")
NORM_WIDTH = 2.0

# per-column (physical span, unit label). Angle span: default full turn in
# radians; ADJUST if source uses degrees or [-pi,pi]. Check observation.py.
COLS = [
    ("angle",        2*np.pi, "rad"),     # <-- verify angle normalization in source
    ("power",        80.0,    "dB"),      # (-100,-20)
    ("SNR",          50.0,    "dB"),      # (-10,40)
    ("noise-ctx",    80.0,    "dB"),      # (-100,-20)
    ("thru-ingress", 38.0,    "dB-bits"), # (30,68) log-throughput
    ("thru-egress",  38.0,    "dB-bits"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="/cluster/scratch/eguzys/pisa_data/user_sets.npz")
    p.add_argument("--encoder",
                   default="/cluster/scratch/eguzys/pisa_data/scaling_out/deepsets_h64_o8.pt")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--output", type=int, default=8)
    p.add_argument("--cap", type=int, default=200000)
    p.add_argument("--out", default="/cluster/scratch/eguzys/pisa_data/per_feature_recon.json")
    a = p.parse_args()

    d = np.load(a.npz)
    X = d[NPZ_X_KEY].astype(np.float32)
    M = d[NPZ_M_KEY].astype(np.float32)
    if a.cap and X.shape[0] > a.cap:
        X, M = X[:a.cap], M[:a.cap]
    n, max_n, dim = X.shape
    assert dim == 6

    model = DeepSetsAE(dim, a.hidden, a.output, max_n)
    sd = torch.load(a.encoder, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()

    err = np.empty_like(X)
    bs = 8192
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(X[i:i+bs]).float()
            mb = torch.from_numpy(M[i:i+bs]).float()
            recon, _ = model(xb, mb)
            err[i:i+bs] = recon.numpy() - X[i:i+bs]

    m = M[:, :, None]
    denom = m.sum(axis=(0, 1)).clip(min=1.0)

    # natural std (spread) per col, over real tokens
    mean = (X * m).sum(axis=(0, 1)) / denom
    std_norm = np.sqrt((((X - mean) ** 2) * m).sum(axis=(0, 1)) / denom)

    # reconstruction RMS per col
    rms_norm = np.sqrt(((err ** 2) * m).sum(axis=(0, 1)) / denom)

    print("\n" + "=" * 82)
    print(f"{'col':>3} {'feature':>13} {'nat.std':>10} {'recon.err':>11} "
          f"{'unit':>9} {'err/std':>9}")
    print(f"{'':>3} {'':>13} {'(physical)':>10} {'(physical)':>11}")
    print("-" * 82)

    rows = []
    for c, (name, span, unit) in enumerate(COLS):
        std_phys = float(std_norm[c]) / NORM_WIDTH * span
        rms_phys = float(rms_norm[c]) / NORM_WIDTH * span
        ratio = rms_phys / std_phys if std_phys > 0 else float("nan")
        print(f"{c:>3} {name:>13} {std_phys:>10.3f} {rms_phys:>11.3f} "
              f"{unit:>9} {ratio:>9.3f}")
        rows.append({"col": c, "feature": name, "unit": unit,
                     "natural_std_physical": std_phys,
                     "recon_err_physical": rms_phys,
                     "err_over_std": ratio,
                     "natural_std_norm": float(std_norm[c]),
                     "recon_err_norm": float(rms_norm[c])})

    print("=" * 82)
    print("\nRead as: 'feature naturally swings by <nat.std> <unit>; the encoder")
    print("blurs each reading by <recon.err> <unit>' (err/std = fraction of the")
    print("natural spread lost to compression). Descriptive, not a verdict.\n")

    with open(a.out, "w") as f:
        json.dump({"encoder": a.encoder, "hidden": a.hidden, "output": a.output,
                   "columns": rows,
                   "note": "physical-unit per-feature recon vs natural spread; "
                           "angle span assumed 2*pi rad — VERIFY against source"},
                  f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
