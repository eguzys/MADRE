import argparse, json, numpy as np, torch
from compare_encoders import DeepSetsAE, masked_recon_loss

p = argparse.ArgumentParser()
p.add_argument("--npz", default="/cluster/scratch/eguzys/pisa_data/user_sets.npz")
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--output", type=int, default=8)
p.add_argument("--epochs", type=int, default=400)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--cap", type=int, default=0, help="0 = use all samples")
p.add_argument("--ep-rows", type=int, default=5005, help="rows per episode (1001 steps x 5 drones)")
p.add_argument("--out", default="/cluster/scratch/eguzys/pisa_data/deepsets_h64_o8.pt")
a = p.parse_args()

d = np.load(a.npz)
# Pin the keys. The old sniffer ("first key containing 'mask'") breaks on the
# merged OMNeT npz, which also carries event_tokens / event_masks / episode_id.
Xn, Mn = d["tokens"], d["masks"]
if a.cap:
    Xn, Mn = Xn[:a.cap], Mn[:a.cap]
X = torch.tensor(Xn, dtype=torch.float32)
M = torch.tensor(Mn, dtype=torch.float32)
dim, max_n = X.shape[2], X.shape[1]

# EPISODE-ALIGNED split. A random-row split LEAKS: step 437 and 438 are
# near-identical, so val recon looks great and means nothing.
# If the npz carries episode_id (OMNeT collections do), split on it EXACTLY --
# no row-arithmetic to misconfigure. Fall back to --ep-rows for old PADRE npz.
if "episode_id" in d:
    epid = d["episode_id"]
    eps = np.unique(epid)
    n_val_ep = max(1, len(eps) // 10)
    val_eps = eps[-n_val_ep:]                      # trailing whole episodes
    is_val = np.isin(epid, val_eps)
    Xtr, Mtr = X[~is_val], M[~is_val]
    Xval, Mval = X[is_val], M[is_val]
    n_ep = len(eps)
    print(f"episode-aligned split VIA episode_id: {n_ep} eps, {n_val_ep} held out",
          flush=True)
else:
    EP_ROWS = a.ep_rows
    n_ep = X.shape[0] // EP_ROWS
    n_val_ep = max(1, n_ep // 10)
    cut = (n_ep - n_val_ep) * EP_ROWS
    Xtr, Mtr = X[:cut], M[:cut]
    Xval, Mval = X[cut:], M[cut:]
    print(f"episode-aligned split VIA --ep-rows={EP_ROWS}: {n_ep} eps, "
          f"{n_val_ep} held out", flush=True)
print(f"  {Xtr.shape[0]} train rows / {Xval.shape[0]} val rows", flush=True)
print(f"tokens {tuple(X.shape)}  dim={dim} max_n={max_n}  "
      f"hidden={a.hidden} output={a.output}", flush=True)

model = DeepSetsAE(dim, a.hidden, a.output, max_n)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val = float("inf")
for ep in range(a.epochs):
    model.train()
    idx = torch.randperm(Xtr.shape[0])
    for i in range(0, Xtr.shape[0], a.batch_size):
        b = idx[i:i+a.batch_size]
        opt.zero_grad()
        out, _ = model(Xtr[b], Mtr[b])
        masked_recon_loss(out, Xtr[b], Mtr[b]).backward()
        opt.step()
    if ep % 10 == 0 or ep == a.epochs - 1:
        model.eval()
        with torch.no_grad():
            vout, _ = model(Xval, Mval)
            vloss = masked_recon_loss(vout, Xval, Mval).item()
        print(f"  epoch {ep:4d}  val_recon={vloss:.4f}", flush=True)
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), a.out)

cfg = {"dim": dim, "hidden": a.hidden, "output": a.output, "max_n": max_n,
       "best_val_recon": best_val, "class": "DeepSetsAE"}
json.dump(cfg, open(a.out.replace(".pt", "_cfg.json"), "w"), indent=2)
print(f"saved best (val_recon={best_val:.4f}) -> {a.out}", flush=True)
