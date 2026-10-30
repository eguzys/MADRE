import torch, numpy as np, importlib.util, json, argparse
from scipy.stats import wasserstein_distance

p = argparse.ArgumentParser()
p.add_argument("--npz",     default="/cluster/scratch/eguzys/pisa_data/user_sets.npz")
p.add_argument("--encoder", default="/cluster/scratch/eguzys/pisa_data/scaling_out/deepsets_h64_o8.pt")
p.add_argument("--hidden",  type=int, default=64)
p.add_argument("--output",  type=int, default=8)
p.add_argument("--cap",     type=int, default=120000)   # held-out slice
p.add_argument("--out",     default="/cluster/scratch/eguzys/pisa_data/distributional_recon.json")
a = p.parse_args()

spec = importlib.util.spec_from_file_location(
    "ce", "/cluster/scratch/eguzys/compare_encoders_fixed.py")
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)

m = ce.DeepSetsAE(dim=6, hidden=a.hidden, output=a.output, max_n=22)
ck = torch.load(a.encoder, map_location="cpu")
m.load_state_dict(ck.get("state_dict", ck) if isinstance(ck, dict) else ck)
m.eval()

d = np.load(a.npz)
X = torch.tensor(d["tokens"][-a.cap:], dtype=torch.float32)   # LAST slice = held-out tail
M = torch.tensor(d["masks"][-a.cap:],  dtype=torch.bool)
print("eval on", X.shape[0], "sets")

COLS = ["angle", "power", "SNR", "noise_ctx", "thru_in", "thru_out"]

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
        out, _ = m(xb, mb)
        for b in range(xb.shape[0]):
            valid = mb[b]
            k = int(valid.sum())
            if k == 0:
                continue
            o = xb[b][valid].numpy()   # (k,6)
            r = out[b][valid].numpy()  # (k,6) decoded, same slots
            for c in range(6):
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

json.dump(res, open(a.out, "w"), indent=2)
print("\nwrote", a.out)
print("\nREAD: ident% high + globW%/setW% low  => identity lost, mass preserved (metric was wrong).")
print("      ident% high + globW%/setW% ALSO high => real capacity loss (comforting story fails).")
