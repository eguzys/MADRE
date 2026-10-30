#!/usr/bin/env python3
"""Train THE frozen shared user-set encoder for the OMNeT track.

One encoder, reconstruction-trained, then frozen and handed to every
architecture so that any remaining ST/deepsets/FFN difference is provably the
Level-2 aggregator and not the Level-1 encoding.

WHY THIS IS NOT src/encoder_experiments/train_final_encoder.py
--------------------------------------------------------------
That script minimises a PLAIN masked MSE over all columns. Measured on real
greedy OMNeT data, the realised per-column std is:

    id_tag_0..3 (SHA256 hash)  0.46-0.71   <- 68% of all variance
    angle (geometry)           0.504
    thru_valid                 0.409
    throughput                 0.389
    interference               0.232
    rssi                       0.171       <- 1.4% of variance
    rssi_valid/noise_floor/
    id_valid/present           0.000       <- dead constants

so a plain MSE spends ~2/3 of an 8-dim bottleneck memorising a deterministic
hash of (episode_seed, user_index) -- incompressible by construction, useless
to a permutation-invariant sum -- and almost ignores signal strength. Every
column is already bounded to [-1,1] by the normalisation in
build_raw_user_tokens, but BOUNDED IS NOT SCALE-MATCHED: an MSE weights each
column by its realised VARIANCE, not its range.

So: the encoder still SEES all 13 columns (identical input to the learned
encoder, so the frozen and learned tracks stay comparable), but the loss is
weighted per column. The weighting is an explicit, reported decision.

USAGE
-----
  # train on greedy, report per-column R^2 on held-out greedy
  python train_frozen_omnet_encoder.py --npz 'frozen_enc/greedy/*.npz' \
      --out enc_greedy_s0.pt --seed 0

  # coverage study: train on greedy, ALSO score held-out random states
  python train_frozen_omnet_encoder.py --npz 'frozen_enc/greedy/*.npz' \
      --eval-npz 'frozen_enc/random/*.npz' --out enc_greedy_s0.pt --seed 0

The saved .pt is a plain state_dict whose encoder-half keys
(phi.0.*, phi.2.*, to_latent.*) are exactly what
MANET/frozen_encoder.FrozenOMNeTUserEncoder reads. A sibling <out>_cfg.json
records the column weights, the realised stds and the val metrics, so a frozen
run can always be traced back to how its encoder was fitted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Authoritative column order: build_raw_user_tokens()'s row[] assignments.
COL_NAMES = [
    "angle", "rssi", "rssi_valid", "interference", "noise_floor",
    "throughput", "thru_valid",
    "id_tag_0", "id_tag_1", "id_tag_2", "id_tag_3", "id_valid", "present",
]
IDTAG_COLS = [7, 8, 9, 10]
# std below this counts as a dead constant: nothing to reconstruct, and 1/var
# weighting would explode. present/id_valid/rssi_valid are exactly 1.0 on every
# sensed row by construction.
CONST_STD_EPS = 1e-6


class DeepSetsAE(nn.Module):
    """phi -> masked sum-pool -> bottleneck -> decoder.

    Identical topology to encoder_experiments/compare_encoders.py's DeepSetsAE
    (and therefore to what FrozenOMNeTUserEncoder reloads); duplicated here
    only so this script has no import dependency on that sweep module.
    """

    def __init__(self, dim, hidden, output, max_n):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_latent = nn.Linear(hidden, output)
        self.dec = nn.Sequential(
            nn.Linear(output, hidden), nn.ReLU(), nn.Linear(hidden, max_n * dim),
        )
        self.max_n, self.dim = max_n, dim

    def forward(self, x, mask):
        h = self.phi(x) * mask.unsqueeze(-1)   # zero padded elements
        z = self.to_latent(h.sum(1))           # permutation-invariant sum
        out = self.dec(z).view(x.shape[0], self.max_n, self.dim)
        return out, z


def load_shards(pattern):
    """Concatenate every npz matching `pattern`, keeping episode_id GLOBALLY
    unique across shards.

    Shards are collected with disjoint --seed blocks so their episode_ids
    already differ, but that is a property of how they were launched, not an
    invariant of this function -- so it is asserted rather than assumed. A
    collision would silently put the same episode on both sides of the
    train/val split.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no npz matched {pattern!r}")
    toks, masks, epids, policies = [], [], [], set()
    seen = set()
    for p in paths:
        d = np.load(p, allow_pickle=True)
        if "episode_id" not in d:
            raise SystemExit(
                f"{p} has no episode_id -- collected by a pre-2026-08-18 "
                "collector. The episode-aligned split cannot be trusted "
                "without it (train_final_encoder.py's --ep-rows fallback "
                "assumes FIXED rows/episode, which OMNeT violates when a "
                "drone dies mid-episode). Re-collect.")
        eid = d["episode_id"]
        dup = seen & set(np.unique(eid).tolist())
        if dup:
            raise SystemExit(f"{p}: episode_id collision with an earlier "
                             f"shard: {sorted(dup)[:5]}")
        seen |= set(np.unique(eid).tolist())
        toks.append(d["tokens"]); masks.append(d["masks"]); epids.append(eid)
        if "policy" in d:
            policies.add(str(d["policy"]))
    X = np.concatenate(toks, 0)
    M = np.concatenate(masks, 0)
    E = np.concatenate(epids, 0)
    print(f"loaded {len(paths)} shard(s) from {pattern!r}: "
          f"{X.shape[0]} rows, {len(np.unique(E))} episodes, "
          f"policies={sorted(policies) or ['?']}", flush=True)
    return X, M, E


def realised_std(X, M):
    """Per-column std over SENSED rows only. Padding rows are all-zero by
    construction and would drag every std toward 0 if included."""
    m = M.astype(bool)
    return np.array([X[:, :, c][m].std() for c in range(X.shape[2])],
                    dtype=np.float64)


def build_col_weights(std, scheme):
    """Per-column loss weights. Returned weights multiply the SQUARED error.

    equal / std : w_c = 1/var_c on informative columns, so every informative
                  column contributes equally to the loss regardless of how
                  loud it is. Dead-constant columns get 0 (nothing to fit).
    drop_idtag  : same, but the 4 SHA256 hash columns are also zeroed. They are
                  incompressible by construction and a permutation-invariant
                  SUM cannot use per-user identity anyway.
    none        : plain MSE -- the naive baseline, kept so the effect of the
                  weighting is measurable rather than asserted.
    """
    w = np.ones_like(std)
    if scheme == "none":
        return w
    var = std ** 2
    w = np.where(std > CONST_STD_EPS, 1.0 / np.maximum(var, 1e-12), 0.0)
    if scheme == "drop_idtag":
        w[IDTAG_COLS] = 0.0
    # Normalise so the loss magnitude stays comparable across schemes.
    nz = w > 0
    if nz.any():
        w = w / w[nz].mean()
    return w


def weighted_recon_loss(out, x, mask, wcol):
    m = mask.unsqueeze(-1)
    se = ((out - x) ** 2) * wcol.view(1, 1, -1)
    return (se * m).sum() / (m.sum().clamp(min=1) * float(wcol.sum().clamp(min=1e-9)))


@torch.no_grad()
def per_column_r2(model, X, M, Y, MY, bs=4096):
    """R^2 per column over SENSED rows: 1 - SSE/SST, SST about the column mean.

    This is the metric that makes "the encoder learned geometry, not the hash"
    checkable instead of asserted. A column the encoder ignores lands at ~0
    (or negative); a constant column is reported as nan and skipped.
    """
    dim = X.shape[2]
    sse = np.zeros(dim); n = 0
    sums = np.zeros(dim); sumsq = np.zeros(dim)
    for i in range(0, X.shape[0], bs):
        xb, mb = X[i:i + bs], M[i:i + bs]
        yb, myb = Y[i:i + bs], MY[i:i + bs]
        out, _ = model(xb, mb)          # ENCODE the raw set...
        m = myb.bool()                  # ...SCORE against the target
        for c in range(dim):
            xc = yb[:, :, c][m].cpu().numpy()
            oc = out[:, :, c][m].cpu().numpy()
            sse[c] += float(((xc - oc) ** 2).sum())
            sums[c] += float(xc.sum()); sumsq[c] += float((xc ** 2).sum())
        n += int(m.sum())
    mean = sums / max(n, 1)
    sst = sumsq - 2 * mean * sums + n * mean ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - sse / sst
    r2[sst <= 1e-9] = np.nan          # constant column: R^2 undefined
    return r2, sse / max(n, 1)


def sorted_targets(X, M):
    """Per-column ASCENDING order statistics of the SENSED values, left-packed.

    The row-indexed target is MIS-SPECIFIED for this encoder (it overfits,
    with best val at epoch 0). The encoder is a masked SUM -- permutation-invariant by
    construction -- so its latent cannot possibly say *which user sits in row
    7*. Row<->value assignment is episode-specific (the layout is seeded per
    episode), so the only way to fit a row-indexed target is to memorise each
    training episode, which cannot transfer. Train fell, val rose, from epoch 0.

    Sorting each column's sensed values turns the target into the column's
    EMPIRICAL DISTRIBUTION: a multiset representation, invariant to the input
    ordering, and exactly the information a permutation-invariant policy can
    use downstream. Same tensor shape as before, so the decoder is unchanged.

    Padding rows are left at 0 and excluded by the returned mask, which is the
    original mask left-packed (k sensed users -> first k rows valid).
    """
    B, U, C = X.shape
    m = M.astype(bool)
    Y = np.zeros_like(X)
    Mo = np.zeros_like(M)
    for i in range(B):
        k = int(m[i].sum())
        if k == 0:
            continue
        vals = X[i][m[i]]                      # (k, C)
        Y[i, :k] = np.sort(vals, axis=0)       # each column sortedly
        Mo[i, :k] = 1.0
    return Y, Mo


@torch.no_grad()
def latent_stats(model, X, M, bs=4096):
    """Distribution of the UNSCALED 8-dim latent over a dataset.

    FrozenOMNeTUserEncoder divides its latent by `latent_scale` from the
    sibling _cfg.json and REFUSES to default it. The latent is a network
    activation, unbounded by
    construction, and it lands in the drone token beside columns that are all
    clipped to [-1,1] -- with `_TOKEN_ABSMAX = 10.0` enforced and an explicit
    "do NOT raise it to make a new column fit, normalise the column".

    So the scale is fitted here, not guessed: latents must come out O(1), the
    same size as their neighbours in the token. Bounding is not scale-matching,
    and that applies to the encoder's own output too.
    """
    zs = []
    for i in range(0, X.shape[0], bs):
        _, z = model(X[i:i + bs], M[i:i + bs])
        zs.append(z.cpu().numpy())
    Z = np.concatenate(zs, 0)
    absZ = np.abs(Z)
    return {
        "p99_9": float(np.percentile(absZ, 99.9)),
        "absmax": float(absZ.max()),
        "per_dim_p99": [float(v) for v in np.percentile(absZ, 99, axis=0)],
    }


# Fixed, deliberately NOT tied to --seed: every arm and every init seed must
# validate on the SAME held-out episodes, so their val losses are comparable
# and the coverage study's seed-spread noise floor contains no split noise.
SPLIT_SEED = 12345


def episode_split(E, val_frac=0.1):
    """Hold out WHOLE episodes, chosen pseudo-randomly at the EPISODE level.

    Whole episodes, because a random-ROW split leaks badly: steps 437 and 438
    of one episode are near-identical, so val recon looks excellent and means
    nothing.

    Not the TRAILING episodes: when two policies are mixed (greedy occupies
    episode_id 0..239 and random 500..619), the trailing 10% of the sorted
    union would be all random episodes, and the arms' val losses would measure
    different things.

    Episode order carries no temporal meaning here (shards come from disjoint
    seed blocks), so a pseudo-random episode-level holdout is valid and, being
    stratified across whatever policies are present, comparable.
    """
    eps = np.unique(E)
    if len(eps) < 2:
        raise SystemExit(
            f"only {len(eps)} episode(s) in the split set -- an episode-aligned "
            "holdout needs at least 2. Collect more shards.")
    n_val = max(1, int(round(len(eps) * val_frac)))
    val_eps = np.random.default_rng(SPLIT_SEED).choice(eps, size=n_val,
                                                       replace=False)
    is_val = np.isin(E, val_eps)
    return ~is_val, is_val, len(eps), n_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="glob for TRAIN shards")
    ap.add_argument("--eval-npz", default=None,
                    help="glob for an OUT-OF-DISTRIBUTION eval set (the "
                         "coverage study: train on greedy, score on random)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--output", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0,
                    help="INIT seed. Run >=3 of these: the spread across them "
                         "is the noise floor the coverage study compares "
                         "distribution shift against.")
    ap.add_argument("--target", choices=["sorted", "rows"], default="sorted",
                    help="sorted (default): reconstruct each column's ASCENDING "
                         "order statistics, i.e. the set's empirical "
                         "distribution -- permutation-invariant, matching the "
                         "encoder. rows: the original row-indexed target, kept "
                         "only to reproduce the S5.3 failure.")
    ap.add_argument("--col-weight",
                    choices=["std", "drop_idtag", "none"], default="std")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)

    Xn, Mn, En = load_shards(a.npz)
    dim, max_n = Xn.shape[2], Xn.shape[1]
    if dim != len(COL_NAMES):
        print(f"!! token width {dim} != {len(COL_NAMES)} known column names -- "
              "the layout changed, re-check build_raw_user_tokens", flush=True)

    # ⚠️ The sorted array is the TARGET ONLY. It must never be fed to the
    # encoder: sorting is per-column and independent, so row i of the sorted
    # array is a Frankenstein (i-th smallest angle, i-th smallest rssi, ...)
    # that corresponds to no real user. phi() is applied per ROW, so encoding
    # it would produce a different latent than the raw set does at inference.
    if a.target == "sorted":
        Yn, MYn = sorted_targets(Xn, Mn)
        print("target=sorted: encoder consumes the RAW set; loss scores the "
              "per-column order statistics (the set's empirical distribution). "
              "See sorted_targets() for why a row-indexed target cannot "
              "generalise across episodes.", flush=True)
    else:
        Yn, MYn = Xn, Mn

    # std of the TARGET (that is what the loss weights).
    std = realised_std(Yn, MYn)
    wcol_np = build_col_weights(std, a.col_weight)
    print(f"\n=== realised per-column std / loss weight "
          f"(scheme={a.col_weight}) ===")
    var = std ** 2
    tot_var = var.sum()
    for c in range(dim):
        nm = COL_NAMES[c] if c < len(COL_NAMES) else f"col{c}"
        share = 100.0 * var[c] / tot_var if tot_var > 0 else 0.0
        print(f"  {nm:12s} std={std[c]:7.4f}  var_share={share:5.1f}%  "
              f"w={wcol_np[c]:8.3f}")

    tr, va, n_ep, n_val_ep = episode_split(En)
    print(f"\nepisode-aligned split: {n_ep} episodes, {n_val_ep} held out "
          f"-> {tr.sum()} train / {va.sum()} val rows", flush=True)

    X = torch.tensor(Xn, dtype=torch.float32, device=dev)
    M = torch.tensor(Mn, dtype=torch.float32, device=dev)
    Y = torch.tensor(Yn, dtype=torch.float32, device=dev)
    MY = torch.tensor(MYn, dtype=torch.float32, device=dev)
    wcol = torch.tensor(wcol_np, dtype=torch.float32, device=dev)
    Xtr, Mtr, Xva, Mva = X[tr], M[tr], X[va], M[va]
    Ytr, MYtr, Yva, MYva = Y[tr], MY[tr], Y[va], MY[va]

    model = DeepSetsAE(dim, a.hidden, a.output, max_n).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n = Xtr.shape[0]
    best = float("inf"); best_sd = None; best_ep = -1
    for ep in range(a.epochs):
        model.train()
        idx = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, a.batch_size):
            b = idx[i:i + a.batch_size]
            out, _ = model(Xtr[b], Mtr[b])
            loss = weighted_recon_loss(out, Ytr[b], MYtr[b], wcol)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vout, _ = model(Xva, Mva)
            vl = weighted_recon_loss(vout, Yva, MYva, wcol).item()
        if vl < best:
            best = vl
            best_ep = ep
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
        if ep % max(1, a.epochs // 10) == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:4d}  train={tot/n:.5f}  val={vl:.5f}"
                  f"{'  *' if vl <= best else ''}", flush=True)

    model.load_state_dict(best_sd)
    model.eval()
    # Print the saved model's identity explicitly: the saved model is the
    # best-val epoch, not the last epoch, and the per-epoch log alone does
    # not say so.
    print(f"\n=== SAVED MODEL = BEST-VAL: epoch {best_ep}, "
          f"val={best:.5f} (final epoch val was {vl:.5f}) ===", flush=True)

    r2_in, mse_in = per_column_r2(model, Xva, Mva, Yva, MYva)
    print(f"\n=== per-column R^2 on held-out IN-DISTRIBUTION states ===")
    for c in range(dim):
        nm = COL_NAMES[c] if c < len(COL_NAMES) else f"col{c}"
        v = r2_in[c]
        print(f"  {nm:12s} R2={'   n/a (const)' if np.isnan(v) else f'{v:+8.4f}'}"
              f"   mse={mse_in[c]:.5f}")

    ood = {}
    if a.eval_npz:
        Xo, Mo, Eo = load_shards(a.eval_npz)
        Yo, MYo = sorted_targets(Xo, Mo) if a.target == "sorted" else (Xo, Mo)
        Xo_t = torch.tensor(Xo, dtype=torch.float32, device=dev)
        Mo_t = torch.tensor(Mo, dtype=torch.float32, device=dev)
        Yo_t = torch.tensor(Yo, dtype=torch.float32, device=dev)
        MYo_t = torch.tensor(MYo, dtype=torch.float32, device=dev)
        with torch.no_grad():
            ol = weighted_recon_loss(model(Xo_t, Mo_t)[0], Yo_t, MYo_t, wcol)
        r2_o, mse_o = per_column_r2(model, Xo_t, Mo_t, Yo_t, MYo_t)
        print(f"\n=== COVERAGE STUDY: per-column R^2 on OUT-OF-DISTRIBUTION "
              f"states ({a.eval_npz}) ===")
        print(f"  weighted val loss  in-dist={best:.5f}   ood={ol.item():.5f}"
              f"   ratio={ol.item()/max(best,1e-12):.3f}")
        for c in range(dim):
            nm = COL_NAMES[c] if c < len(COL_NAMES) else f"col{c}"
            vi, vo = r2_in[c], r2_o[c]
            if np.isnan(vi) and np.isnan(vo):
                continue
            print(f"  {nm:12s} in={vi:+8.4f}  ood={vo:+8.4f}  "
                  f"drop={vi - vo:+8.4f}")
        ood = {"ood_glob": a.eval_npz, "ood_loss": ol.item(),
               "ood_loss_ratio": ol.item() / max(best, 1e-12),
               "ood_r2": [None if np.isnan(v) else float(v) for v in r2_o]}

    lat = latent_stats(model, X, M)
    # 1.5x headroom over the training absmax: the live distribution can exceed
    # the fitted one (the OLD encoder measured 21.1 peak on a random drive
    # against a 67.57 training p99.9, i.e. it UNDERSHOT -- but the reverse is
    # the dangerous direction and costs nothing to guard).
    latent_scale = float(max(lat["absmax"] * 1.5, 1e-6))
    print(f"\n=== latent scale fit (unscaled |z| over the training set) ===")
    print(f"  p99.9={lat['p99_9']:.4f}  absmax={lat['absmax']:.4f}"
          f"  -> latent_scale={latent_scale:.4f}")
    print(f"  scaled |z| absmax = {lat['absmax']/latent_scale:.4f}"
          f"   (must be << _TOKEN_ABSMAX=10.0, ideally ~O(1))")
    print(f"  per-dim p99: {[round(v,3) for v in lat['per_dim_p99']]}")
    if a.eval_npz:
        lat_o = latent_stats(model, Xo_t, Mo_t)
        print(f"  OOD absmax={lat_o['absmax']:.4f}  -> scaled "
              f"{lat_o['absmax']/latent_scale:.4f}"
              f"{'   ** EXCEEDS 1.0, latent_scale undershoots OOD **' if lat_o['absmax']/latent_scale > 1.0 else ''}")
        ood["ood_latent"] = lat_o

    torch.save(model.state_dict(), a.out)
    cfg = {
        # --- consumed by MANET/frozen_encoder.FrozenOMNeTUserEncoder ---
        "latent_scale": latent_scale,
        "latent_scale_basis":
            f"1.5x absmax over the TRAIN set of {a.npz}: p99.9="
            f"{lat['p99_9']:.4f}, absmax={lat['absmax']:.4f}; global, not "
            f"per-dim. Chosen so the latent is O(1) like every other token "
            f"column (_TOKEN_ABSMAX=10.0 is a ceiling, not a target).",
        "latent_p99_9": lat["p99_9"], "latent_absmax": lat["absmax"],
        "latent_per_dim_p99": lat["per_dim_p99"],
        "hidden": a.hidden, "output": a.output,
        "token_dim": dim, "max_users": max_n, "class": "DeepSetsAE",
        "train_glob": a.npz, "dim": dim, "max_n": max_n,
        "hidden": a.hidden, "out_dim": a.output,
        "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
        "seed": a.seed, "col_weight_scheme": a.col_weight,
        "target": a.target,
        "col_names": COL_NAMES[:dim],
        "realised_std": [float(v) for v in std],
        "col_weights": [float(v) for v in wcol_np],
        "n_episodes": int(n_ep), "n_val_episodes": int(n_val_ep),
        "best_val_weighted_loss": float(best),
        "best_epoch": int(best_ep),
        "val_r2": [None if np.isnan(v) else float(v) for v in r2_in],
        **ood,
    }
    with open(os.path.splitext(a.out)[0] + "_cfg.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nsaved -> {a.out}")
    print(f"saved -> {os.path.splitext(a.out)[0]}_cfg.json")


if __name__ == "__main__":
    main()
