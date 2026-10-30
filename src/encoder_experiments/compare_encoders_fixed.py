import argparse, json, os, numpy as np, torch, torch.nn as nn

class DeepSetsAE(nn.Module):
    def __init__(self, dim, hidden, output, max_n):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.to_latent = nn.Linear(hidden, output)
        self.dec = nn.Sequential(nn.Linear(output, hidden), nn.ReLU(), nn.Linear(hidden, max_n*dim))
        self.max_n, self.dim = max_n, dim
    def forward(self, x, mask):
        h = self.phi(x) * mask.unsqueeze(-1)
        z = self.to_latent(h.sum(1))
        out = self.dec(z).view(x.shape[0], self.max_n, self.dim)
        return out, z

class SetAttnAE(nn.Module):
    def __init__(self, dim, hidden, output, max_n, heads=4):
        super().__init__()
        self.embed = nn.Linear(dim, hidden)
        self.attn  = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.to_latent = nn.Linear(hidden, output)
        self.dec = nn.Sequential(nn.Linear(output, hidden), nn.ReLU(), nn.Linear(hidden, max_n*dim))
        self.max_n, self.dim = max_n, dim
    def forward(self, x, mask):
        h = self.embed(x)
        kpm = (mask == 0)
        empty = kpm.all(dim=1)
        if empty.any():
            kpm = kpm.clone()
            kpm[empty, 0] = False
        a, _ = self.attn(h, h, h, key_padding_mask=kpm)
        a = torch.nan_to_num(a, nan=0.0)
        a = a * mask.unsqueeze(-1)
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        z = self.to_latent(a.sum(1) / denom)
        out = self.dec(z).view(x.shape[0], self.max_n, self.dim)
        return out, z

def masked_recon_loss(out, x, mask):
    m = mask.unsqueeze(-1)
    return (((out - x)**2) * m).sum() / m.sum().clamp(min=1)

def train_eval(model, Xtr, Mtr, Xval, Mval, epochs, bs, save_path, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            out, _ = model(Xtr[b], Mtr[b])
            loss = masked_recon_loss(out, Xtr[b], Mtr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % max(1, epochs//10) == 0 or ep == epochs-1:
            model.eval()
            with torch.no_grad():
                vout, _ = model(Xval, Mval)
                vloss = masked_recon_loss(vout, Xval, Mval).item()
            print(f"    epoch {ep:4d}  val_recon={vloss:.4f}", flush=True)
            if vloss < best_val:
                best_val = vloss
                torch.save(model.state_dict(), save_path)
    return best_val

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="/cluster/scratch/eguzys/pisa_data/user_sets.npz")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--sweep-output", type=int, nargs="+", default=[8,16,24,32])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--out-dir", default="/cluster/scratch/eguzys/pisa_data/sweep_fixed")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d = np.load(a.npz)
    print("npz keys:", list(d.keys()), flush=True)
    tok_key  = [k for k in d.keys() if d[k].ndim==3][0]
    mask_key = [k for k in d.keys() if 'mask' in k.lower()][0]
    X = torch.tensor(d[tok_key], dtype=torch.float32)
    M = torch.tensor(d[mask_key], dtype=torch.float32)
    if a.cap and X.shape[0] > a.cap:
        X, M = X[:a.cap], M[:a.cap]
        print(f"subsampled to {a.cap}", flush=True)
    B, max_n, dim = X.shape
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(B, generator=g)
    n_val = int(B * a.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Mtr = X[tr_idx], M[tr_idx]
    Xval, Mval = X[val_idx], M[val_idx]
    print(f"tokens {tuple(X.shape)}  train {len(tr_idx)}  val {len(val_idx)}  hidden(fixed)={a.hidden}  epochs={a.epochs}\n", flush=True)
    results = {}
    for out in a.sweep_output:
        print(f"########## output width = {out} ##########", flush=True)
        print(f"--- DeepSetsAE ---", flush=True)
        ds = train_eval(DeepSetsAE(dim, a.hidden, out, max_n), Xtr, Mtr, Xval, Mval, a.epochs, a.batch_size, os.path.join(a.out_dir, f"deepsets_h{a.hidden}_o{out}.pt"))
        print(f"--- SetAttnAE ---", flush=True)
        at = train_eval(SetAttnAE(dim, a.hidden, out, max_n), Xtr, Mtr, Xval, Mval, a.epochs, a.batch_size, os.path.join(a.out_dir, f"setattn_h{a.hidden}_o{out}.pt"))
        results[out] = (ds, at)
    print("\n================ SUMMARY (held-out val recon, lower=better) ================", flush=True)
    print(f"{'output':>8} {'DeepSets':>12} {'SetAttn':>12}", flush=True)
    for out,(ds,at) in results.items():
        print(f"{out:>8} {ds:>12.4f} {at:>12.4f}", flush=True)
    summary = {"hidden": a.hidden, "epochs": a.epochs, "val_frac": a.val_frac, "n_train": len(tr_idx), "n_val": len(val_idx), "seed": a.seed, "results_val_recon": {int(o): {"deepsets": ds, "setattn": at} for o,(ds,at) in results.items()}}
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    print(f"\nsaved summary -> {os.path.join(a.out_dir,'summary.json')}", flush=True)
