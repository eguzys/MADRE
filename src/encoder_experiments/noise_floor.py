#!/usr/bin/env python3
"""
noise_floor.py

Thesis argument this supports:
    Shows the frozen h64/o8 DeepSets encoder distorts each observation channel
    less than the reference noise/spread in that channel.

Assumptions:
    - DeepSetsAE importable from compare_encoders_fixed.py, forward(x,mask)->(out,z)
    - npz keys 'tokens' and 'masks' by default
    - column order:
      0 angle, 1 power(faded), 2 SNR(faded), 3 noise-context,
      4 throughput-ingress, 5 throughput-egress
"""

import argparse
import os
import json
import sys
import numpy as np
import torch

SWEEP_SCRIPT_DIR = os.environ.get("SWEEP_SCRIPT_DIR", "/cluster/scratch/eguzys")
sys.path.insert(0, SWEEP_SCRIPT_DIR)

try:
    from compare_encoders_fixed import DeepSetsAE
except Exception as e:
    print(f"FATAL: cannot import DeepSetsAE from {SWEEP_SCRIPT_DIR}: {e}")
    sys.exit(1)

NPZ_X_KEY = os.environ.get("NPZ_X_KEY", "tokens")
NPZ_M_KEY = os.environ.get("NPZ_M_KEY", "masks")

SIGNAL_MINMAX = (-100.0, -20.0)
SNR_MINMAX = (-10.0, 40.0)
NORM_WIDTH = 2.0

COL_NAMES = [
    "angle",
    "power(faded)",
    "SNR(faded)",
    "noise-ctx",
    "thru-ingress",
    "thru-egress",
]

FADED_COLS = {1, 2}


def masked_rms_per_col(err, mask):
    m = mask[:, :, None]
    sq = (err ** 2) * m
    denom = m.sum(axis=(0, 1)).clip(min=1.0)
    return np.sqrt(sq.sum(axis=(0, 1)) / denom)


def masked_std_per_col(X, mask):
    m = mask[:, :, None]
    n = m.sum(axis=(0, 1)).clip(min=1.0)
    mean = (X * m).sum(axis=(0, 1)) / n
    var = (((X - mean) ** 2) * m).sum(axis=(0, 1)) / n
    return np.sqrt(var)


def apply_fading_reference(X, mask, sigma_dB, seed=1000):
    rng = np.random.default_rng(seed)
    n, max_n, _ = X.shape
    fade_dB = rng.normal(0.0, sigma_dB, size=(n, max_n))

    span_pow = SIGNAL_MINMAX[1] - SIGNAL_MINMAX[0]
    span_snr = SNR_MINMAX[1] - SNR_MINMAX[0]

    shift_pow = (fade_dB / span_pow) * NORM_WIDTH
    shift_snr = (fade_dB / span_snr) * NORM_WIDTH

    denom = mask.sum().clip(min=1.0)

    ref_rms = {
        1: float(np.sqrt(((shift_pow ** 2) * mask).sum() / denom)),
        2: float(np.sqrt(((shift_snr ** 2) * mask).sum() / denom)),
    }

    return ref_rms


def norm_to_dB(err_norm, minmax):
    return err_norm / NORM_WIDTH * (minmax[1] - minmax[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="/cluster/scratch/eguzys/pisa_data/user_sets.npz")
    p.add_argument(
        "--encoder",
        default="/cluster/scratch/eguzys/pisa_data/scaling_out/deepsets_h64_o8.pt",
    )
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--output", type=int, default=8)
    p.add_argument("--sigma-dB", type=float, default=4.0)
    p.add_argument("--cap", type=int, default=200000)
    p.add_argument("--out", default="/cluster/scratch/eguzys/pisa_data/noise_floor.json")
    a = p.parse_args()

    device = "cpu"

    d = np.load(a.npz)

    assert NPZ_X_KEY in d and NPZ_M_KEY in d, (
        f"npz keys {list(d.keys())}; expected {NPZ_X_KEY},{NPZ_M_KEY}"
    )

    X = d[NPZ_X_KEY].astype(np.float32)
    M = d[NPZ_M_KEY].astype(np.float32)

    if a.cap and X.shape[0] > a.cap:
        X = X[:a.cap]
        M = M[:a.cap]

    n, max_n, dim = X.shape
    assert dim == 6, f"expected 6 cols, got {dim}"

    print(
        f"tokens {X.shape}  sigma_dB={a.sigma_dB}  encoder={a.encoder}",
        flush=True,
    )

    model = DeepSetsAE(dim, a.hidden, a.output, max_n)

    sd = torch.load(a.encoder, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd)
    model.eval()

    err = np.empty_like(X)
    bs = 8192

    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(X[i:i + bs]).float()
            mb = torch.from_numpy(M[i:i + bs]).float()
            recon, _ = model(xb, mb)
            err[i:i + bs] = recon.numpy() - X[i:i + bs]

    enc_rms = masked_rms_per_col(err, M)
    sig_std = masked_std_per_col(X, M)
    fade_ref = apply_fading_reference(X, M, a.sigma_dB)

    print()
    print("=" * 78)
    print(
        f"{'col':>3} {'channel':>14} {'enc_rms':>9} {'reference':>11} "
        f"{'ref_type':>10} {'ratio':>7} {'verdict':>22}"
    )
    print("-" * 78)

    rows = []

    for c in range(6):
        enc = float(enc_rms[c])

        if c in FADED_COLS:
            ref = fade_ref[c]
            ref_type = "fading"
        else:
            ref = float(sig_std[c])
            ref_type = "signal-std"

        ratio = enc / ref if ref > 0 else float("nan")

        if c in FADED_COLS:
            verdict = "BELOW noise floor" if ratio < 1 else "ABOVE noise floor"
        else:
            verdict = f"{(1 - ratio) * 100:.0f}% of spread kept" if ratio < 1 else "exceeds spread"

        print(
            f"{c:>3} {COL_NAMES[c]:>14} {enc:>9.4f} {ref:>11.4f} "
            f"{ref_type:>10} {ratio:>7.3f} {verdict:>22}"
        )

        rows.append(
            {
                "col": c,
                "channel": COL_NAMES[c],
                "enc_rms": enc,
                "reference": ref,
                "ref_type": ref_type,
                "ratio": ratio,
            }
        )

    print("=" * 78)

    pow_dB = norm_to_dB(float(enc_rms[1]), SIGNAL_MINMAX)
    snr_dB = norm_to_dB(float(enc_rms[2]), SNR_MINMAX)

    print()
    print("QUOTABLE faded channels, in dB:")
    print(
        f"  power channel: encoder distortion = {pow_dB:.2f} dB  "
        f"(injected fading sigma = {a.sigma_dB:.1f} dB)  -> "
        f"{'BELOW' if pow_dB < a.sigma_dB else 'ABOVE'} noise"
    )
    print(
        f"  SNR   channel: encoder distortion = {snr_dB:.2f} dB  "
        f"(injected fading sigma = {a.sigma_dB:.1f} dB)  -> "
        f"{'BELOW' if snr_dB < a.sigma_dB else 'ABOVE'} noise"
    )

    summary = {
        "encoder": a.encoder,
        "sigma_dB": a.sigma_dB,
        "n_tokens": int(n),
        "columns": rows,
        "power_enc_dB": pow_dB,
        "snr_enc_dB": snr_dB,
        "note": (
            "fidelity not task-sufficiency; pair with scaling curve "
            "and frozen-shared structural argument"
        ),
    }

    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
