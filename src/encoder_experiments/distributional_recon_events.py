import torch, numpy as np, importlib.util, json, argparse
from scipy.stats import wasserstein_distance

p = argparse.ArgumentParser()
p.add_argument("--npz",     default="/cluster/scratch/eguzys/pisa_data/user_sets.npz")
p.add_argument("--encoder", default="/cluster/scratch/eguzys/pisa_data/scaling_out/deepsets_h64_o8.pt")
p.add_argument("--hidden",  type=int, default=64)
p.add_argument("--output",  type=int, default=8)
p.add_argument("--cap",     type=int, default=120000)   # held-out slice
p.add_argument("--latent-mode", choices=["real","shuffle","noise"], default="real",
               help="real=normal; shuffle=permute latents across sets within batch "
                    "(input-specificity baseline); noise=gaussian latent (floor)")
p.add_argument("--out",     default="/cluster/scratch/eguzys/pisa_data/distributional_recon.json")
a = p.parse_args()

# Resolve the sibling module from __file__, NOT a hardcoded cluster path,
# so the script works wherever the repository is checked out.
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ce", os.path.join(_HERE, "compare_encoders_fixed.py"))
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)

d0 = np.load(a.npz)
DIM, MAXN = d0["tokens"].shape[2], d0["tokens"].shape[1]
m = ce.DeepSetsAE(dim=DIM, hidden=a.hidden, output=a.output, max_n=MAXN)
ck = torch.load(a.encoder, map_location="cpu")
m.load_state_dict(ck.get("state_dict", ck) if isinstance(ck, dict) else ck)
m.eval()

d = np.load(a.npz)
# EPISODE-ALIGNED held-out split. Rows here are (step x drone): step 437 and
# 438 are near-identical, so a row-count split would leak correlated timesteps
# into the validation score. Split by EPISODE.
epid = d["episode_id"]
eps = np.unique(epid)
n_val = max(1, len(eps) // 10)                      # 10% held out
val_eps = eps[-n_val:]                              # trailing episodes
sel = np.isin(epid, val_eps)
X = torch.tensor(d["tokens"][sel], dtype=torch.float32)
M = torch.tensor(d["masks"][sel],  dtype=torch.bool)
print(f"episode-aligned val: {len(val_eps)}/{len(eps)} episodes held out "
      f"({X.shape[0]} sets)")

# 13-col token (rssi is MEASURED, and rssi_valid sits at idx 2).
COLS = ["snir", "snir_valid", "per", "ber", "size", "is_drop",
        "drop_retry", "drop_queue", "drop_corrupt", "drop_notforus",
        "from_drone", "present"]

# Token-schema drift guard: COLS must name every column of the token.
# Was a silent IndexError (c in range(DIM), COLS too short) when the token
# grew 12->13 (measured rssi_valid, OMNET.md S5.2) and this list didn't.
assert len(COLS) == DIM, (
    f"COLS has {len(COLS)} names but token DIM={DIM} -- token schema "
    f"changed (OMNET.md S5.2) and COLS was not updated. COLS={COLS}")

# SELECT ON THE FIVE PHYSICAL CHANNELS ONLY (OMNET.md S10).
#   EXCLUDE id_tag_0..3 -- random by construction; its recon error is meaningless
#           noise, and averaging it in flatters or punishes at random.
#   EXCLUDE *_valid / present -- near-constant; they reconstruct "perfectly" and
#           drag any aggregate toward zero.
# This is the OMNeT analogue of PADRE's noise_ctx-dominates-MSE lesson.
# Measured on events_cut.npz (614k cut events): keep the channels with real
# spread, drop the degenerate ones.
#   ber            identically 0 (INET ReceiverBase.cc:112, by design)
#   drop_notforus  identically 0 BY CONSTRUCTION (this is the CUT set)
#   drop_queue     0.1% nonzero -- effectively constant
#   present        constant 1.0 within the mask
#   snir_valid     collinear with snir nonzero% (66.0 both) -- redundant flag
#   per/drop_corrupt ~2% nonzero -- rare-but-real; EXCLUDED as near-constant
#     (same principle as the user token's valid/present exclusion)
PHYSICAL = ["snir", "size", "from_drone", "is_drop", "drop_retry"]
PHYS_IDX = [COLS.index(c) for c in PHYSICAL]

# accumulate per-feature: pooled valid ORIGINAL vals, pooled valid DECODED vals,
# per-set identity error (matched slot i vs slot i, valid only)
orig_vals = [[] for _ in COLS]
dec_vals  = [[] for _ in COLS]
id_err    = [[] for _ in COLS]   # |orig_i - dec_i| over valid slots
# per-set distributional error (wasserstein between the two clouds, per set, per feature)
set_wass  = [[] for _ in COLS]

bs = 4096
with torch.no_grad():
    for s in range(0, X.shape[0], bs):
        xb, mb = X[s:s+bs], M[s:s+bs]
        if a.latent_mode == "real":
            out, _ = m(xb, mb)
        else:
            h = m.phi(xb) * mb.unsqueeze(-1)
            z = m.to_latent(h.sum(1))
            if a.latent_mode == "shuffle":
                perm = torch.randperm(z.shape[0])
                z = z[perm]
            else:
                z = torch.randn_like(z) * z.std() + z.mean()
            out = m.dec(z).view(xb.shape[0], m.max_n, m.dim)
        for b in range(xb.shape[0]):
            valid = mb[b]
            k = int(valid.sum())
            if k == 0:
                continue
            o = xb[b][valid].numpy()   # (k,6)
            r = out[b][valid].numpy()  # (k,6) decoded, same slots
            for c in range(DIM):
                orig_vals[c].append(o[:, c])
                dec_vals[c].append(r[:, c])
                id_err[c].append(np.abs(o[:, c] - r[:, c]))
                if k >= 2:
                    set_wass[c].append(wasserstein_distance(o[:, c], r[:, c]))

res = {}
for c, name in enumerate(COLS):
    ov = np.concatenate(orig_vals[c])
    dv = np.concatenate(dec_vals[c])
    ie = np.concatenate(id_err[c])
    std = float(ov.std())
    # GLOBAL distributional distance: the two pooled clouds across ALL sets
    glob_wass = float(wasserstein_distance(ov, dv))
    res[name] = {
        "nat_std":            std,
        "identity_err_mean":  float(ie.mean()),
        "identity_pct":       float(ie.mean() / std * 100) if std > 0 else None,
        "global_wass":        glob_wass,
        "global_wass_pct":    float(glob_wass / std * 100) if std > 0 else None,
        "per_set_wass_mean":  float(np.mean(set_wass[c])) if set_wass[c] else None,
        "per_set_wass_pct":   float(np.mean(set_wass[c]) / std * 100) if set_wass[c] and std > 0 else None,
    }

print(f"\n{'feature':<10} {'nat_std':>8} {'ident%':>8} {'globW%':>8} {'setW%':>8}")
for name in COLS:
    r = res[name]
    def g(k): return f"{r[k]:>8.1f}" if r[k] is not None else f"{'--':>8}"
    print(f"{name:<10} {r['nat_std']:>8.3f} {g('identity_pct')} {g('global_wass_pct')} {g('per_set_wass_pct')}")

# THE SELECTION METRIC: setW% over the FIVE PHYSICAL CHANNELS ONLY.
# Averaging all 13 would mislead -- id_tag is random by construction (its error is
# meaningless noise) and *_valid/present are near-constant (they reconstruct
# "perfectly" and flatter the mean). Neither says anything about capacity.
phys = [res[COLS[i]]["per_set_wass_pct"] for i in PHYS_IDX
        if res[COLS[i]]["per_set_wass_pct"] is not None]
res["_PHYSICAL_setW_mean"] = float(np.mean(phys))
print(f"\n>>> setW% PHYSICAL MEAN = {np.mean(phys):.2f}   "
      f"({', '.join(PHYSICAL)})")
print(">>> THIS is the number the sweep selects on. Lower = better.")

json.dump(res, open(a.out, "w"), indent=2)
print("\nwrote", a.out)
print("\nREAD: ident% high + globW%/setW% low  => identity lost, mass preserved (metric was wrong).")
print("      ident% high + globW%/setW% ALSO high => real capacity loss (comforting story fails).")
