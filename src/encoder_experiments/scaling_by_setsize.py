#!/usr/bin/env python3
"""
scaling_by_setsize.py

Question this answers (NOT the same as compare_encoders_fixed.py):
    Does DeepSets-AE reconstruction error degrade as the number of nearby
    users (set cardinality k) grows, and does more hidden/output capacity
    keep it flat? This is a *scaling curve*, not a single averaged number.

Method:
    - Train DeepSets AE at each (hidden, output) in a small grid.
    - On the held-out validation split, compute masked reconstruction error
      PER SET, then bucket by k = number of real (unmasked) users in that set.
    - Report mean recon per k-bucket AND the population of each bucket, so
      the sparse large-k tail is visible and not over-trusted.

IMPORTANT CAVEAT (printed at runtime too):
    Collected data is U(5,23); max observed k is ~22. Any statement about
    k=30 is EXTRAPOLATION beyond the data. This script shows the trend up to
    the observed max; it cannot directly measure k=30.

Assumptions to verify (fail loud in <30s if wrong):
    A1. compare_encoders_fixed.py is importable and exposes DeepSetsAE with
        signature DeepSetsAE(dim, hidden, out, max_n).
    A2. npz has arrays keyed 'X' (M, max_n, dim) and 'M' (M, max_n) masks
        (1=real, 0=pad). If your keys differ, set NPZ_X_KEY / NPZ_M_KEY.
    A3. Masked reconstruction error = per-element MSE over real tokens only.
        Matches the training objective in train_eval (verify).
If any assumption is wrong the script asserts early with a clear message.
"""

import argparse, os, json, sys
import numpy as np
import torch
import torch.nn.functional as F

# ---- A1: reuse the exact model class from the sweep script ----
# The sweep script lives loose in scratch; import by path.
SWEEP_SCRIPT_DIR = os.environ.get(
    "SWEEP_SCRIPT_DIR", "/cluster/scratch/eguzys"
)
sys.path.insert(0, SWEEP_SCRIPT_DIR)
try:
    from compare_encoders_fixed import DeepSetsAE  # noqa: E402
except Exception as e:
    print("FATAL: could not import DeepSetsAE from compare_encoders_fixed.py")
    print(f"  looked in: {SWEEP_SCRIPT_DIR}")
    print(f"  error: {e}")
    print("  Fix: set SWEEP_SCRIPT_DIR env var to the script's directory,")
    print("  or check the class name / signature.")
    sys.exit(1)

NPZ_X_KEY = os.environ.get("NPZ_X_KEY", "X")
NPZ_M_KEY = os.environ.get("NPZ_M_KEY", "M")


def masked_recon_per_set(model, X, M, device, bs=4096):
    """Return per-set mean reconstruction error over that set's real tokens.
    Shape in: X (n, max_n, dim), M (n, max_n). Out: (n,) float."""
    model.eval()
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(X[i:i+bs]).float().to(device)
            mb = torch.from_numpy(M[i:i+bs]).float().to(device)
            recon, _ = model(xb, mb)  # forward(x, mask) -> (recon, latent)

            # per-element squared error, then mask, then per-set mean over reals
            se = (recon - xb) ** 2
            se = se.mean(dim=-1)
            se = se * mb
            denom = mb.sum(dim=-1).clamp(min=1.0)
            per_set = se.sum(dim=-1) / denom
            out[i:i+bs] = per_set.cpu().numpy()
    return out


def train_one(model, Xtr, Mtr, device, epochs, bs, lr=1e-3):
    """Minimal training loop mirroring train_eval's masked-MSE objective.
    Kept local so this script is self-contained; verify it matches your
    train_eval loss, i.e. masked per-element MSE.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    idx = np.arange(n)

    for ep in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            xb = torch.from_numpy(Xtr[b]).float().to(device)
            mb = torch.from_numpy(Mtr[b]).float().to(device)
            recon, _ = model(xb, mb)  # forward(x, mask) -> (recon, latent)

            se = ((recon - xb) ** 2).mean(dim=-1) * mb
            loss = se.sum() / mb.sum().clamp(min=1.0)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="/cluster/scratch/eguzys/pisa_data/user_sets.npz")
    p.add_argument("--out-dir", default="/cluster/scratch/eguzys/pisa_data/scaling_out")
    p.add_argument("--cap", type=int, default=200000)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--output", type=int, nargs="+", default=[8, 16])
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    d = np.load(a.npz)
    assert NPZ_X_KEY in d and NPZ_M_KEY in d, (
        f"npz keys are {list(d.keys())}; expected '{NPZ_X_KEY}' and "
        f"'{NPZ_M_KEY}'. Set NPZ_X_KEY / NPZ_M_KEY env vars."
    )

    X = d[NPZ_X_KEY].astype(np.float32)
    M = d[NPZ_M_KEY].astype(np.float32)

    assert X.ndim == 3 and M.ndim == 2, f"bad shapes X{X.shape} M{M.shape}"
    assert X.shape[:2] == M.shape, "X and M set/element dims must match"

    if a.cap and X.shape[0] > a.cap:
        X, M = X[:a.cap], M[:a.cap]

    n, max_n, dim = X.shape
    k_all = M.sum(axis=1).astype(int)
    k_min, k_max = int(k_all.min()), int(k_all.max())

    print("=" * 60)
    print(f"tokens {X.shape}  dim={dim}  max_n={max_n}")
    print(f"observed set-size k: min={k_min}  max={k_max}  mean={k_all.mean():.2f}")
    print(
        f"*** DATA CEILING: max k in data = {k_max}. "
        f"Claims about k>{k_max} are EXTRAPOLATION. ***"
    )
    print("k-bucket populations (val trend only trustworthy where n is large):")

    for k in range(k_min, k_max + 1):
        c = int((k_all == k).sum())
        bar = "#" * min(50, c * 50 // max(1, k_all.size // 10))
        print(f"   k={k:2d}: {c:8d}  {bar}")

    print("=" * 60, flush=True)

    # Split
    perm = np.random.permutation(n)
    n_val = int(n * a.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xtr, Mtr = X[tr_idx], M[tr_idx]
    Xval, Mval = X[val_idx], M[val_idx]
    kval = M[val_idx].sum(axis=1).astype(int)

    results = {}

    for hidden in a.hidden:
        for out in a.output:
            tag = f"h{hidden}_o{out}"
            print(f"\n--- DeepSets {tag} ---", flush=True)

            model = DeepSetsAE(dim, hidden, out, max_n)
            model = train_one(model, Xtr, Mtr, device, a.epochs, a.batch_size)
            per_set = masked_recon_per_set(model, Xval, Mval, device)

            overall = float(per_set.mean())
            by_k = {}

            for k in range(k_min, k_max + 1):
                sel = kval == k
                cnt = int(sel.sum())
                if cnt == 0:
                    continue

                by_k[k] = {
                    "mean_recon": float(per_set[sel].mean()),
                    "n": cnt,
                }

            results[tag] = {
                "overall": overall,
                "by_k": by_k,
                "hidden": hidden,
                "output": out,
            }

            print(f"    overall val_recon = {overall:.4f}")
            print(f"    {'k':>3} {'recon':>8} {'n':>8}")

            for k in sorted(by_k):
                print(
                    f"    {k:>3} {by_k[k]['mean_recon']:>8.4f} "
                    f"{by_k[k]['n']:>8d}"
                )

            torch.save(
                model.state_dict(),
                os.path.join(a.out_dir, f"deepsets_{tag}.pt"),
            )

    summary = {
        "npz": a.npz,
        "cap": a.cap,
        "epochs": a.epochs,
        "k_min": k_min,
        "k_max": k_max,
        "k_mean": float(k_all.mean()),
        "note": (
            f"max observed k={k_max}; claims about larger sets are "
            "extrapolation not measurement"
        ),
        "results": results,
    }

    with open(os.path.join(a.out_dir, "scaling_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nwrote {os.path.join(a.out_dir, 'scaling_summary.json')}")


if __name__ == "__main__":
    main()
