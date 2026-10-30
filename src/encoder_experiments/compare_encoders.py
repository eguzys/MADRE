import argparse, numpy as np, torch, torch.nn as nn

# ---- two-knob encoders. both: input set (B, N, 6) + mask (B, N) -> latent (B, output) ----

class DeepSetsAE(nn.Module):
    """phi -> masked sum-pool -> bottleneck(output) -> rho. Two knobs: hidden, output."""
    def __init__(self, dim, hidden, output, max_n):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.to_latent = nn.Linear(hidden, output)
        self.dec = nn.Sequential(nn.Linear(output, hidden), nn.ReLU(), nn.Linear(hidden, max_n*dim))
        self.max_n, self.dim = max_n, dim
    def forward(self, x, mask):
        h = self.phi(x) * mask.unsqueeze(-1)          # zero padded elements
        z = self.to_latent(h.sum(1))                  # perm-invariant sum
        out = self.dec(z).view(x.shape[0], self.max_n, self.dim)
        return out, z

class SetAttnAE(nn.Module):
    """per-elem embed -> self-attention -> masked mean -> bottleneck(output) -> rho.
       Two knobs: hidden, output. (NO layernorm, to avoid frozen magnitude-stripping.)"""
    def __init__(self, dim, hidden, output, max_n, heads=4):
        super().__init__()
        self.embed = nn.Linear(dim, hidden)
        self.attn  = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.to_latent = nn.Linear(hidden, output)
        self.dec = nn.Sequential(nn.Linear(output, hidden), nn.ReLU(), nn.Linear(hidden, max_n*dim))
        self.max_n, self.dim = max_n, dim
    def forward(self, x, mask):
        h = self.embed(x)
        kpm = (mask == 0)                             # True = ignore
        # guard: a fully-empty set -> all-(-inf) softmax -> nan. unmask row 0 for those.
        empty = kpm.all(dim=1)
        if empty.any():
            kpm = kpm.clone()
            kpm[empty, 0] = False
        a, _ = self.attn(h, h, h, key_padding_mask=kpm)
        a = torch.nan_to_num(a, nan=0.0)              # belt-and-suspenders
        a = a * mask.unsqueeze(-1)
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        z = self.to_latent(a.sum(1) / denom)          # masked mean
        out = self.dec(z).view(x.shape[0], self.max_n, self.dim)
        return out, z

def masked_recon_loss(out, x, mask):
    m = mask.unsqueeze(-1)
    return (((out - x)**2) * m).sum() / m.sum().clamp(min=1)

def train(model, X, M, epochs, bs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            out, _ = model(X[b], M[b])
            loss = masked_recon_loss(out, X[b], M[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % max(1, epochs//10) == 0 or ep == epochs-1:
            print(f"  epoch {ep:4d}  recon={tot/n:.4f}")
    return tot/n

# --- replaces the __main__ block: sweep OUTPUT, fix HIDDEN ---
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="/cluster/scratch/eguzys/pisa_data/user_sets.npz")
    p.add_argument("--hidden", type=int, default=128)          # fixed, large
    p.add_argument("--sweep-output", type=int, nargs="+", default=[8,16,24,32])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args()

    d = np.load(a.npz)
    print("npz keys:", list(d.keys()))
    tok_key  = [k for k in d.keys() if d[k].ndim==3][0]
    mask_key = [k for k in d.keys() if 'mask' in k.lower()][0]
    X = torch.tensor(d[tok_key], dtype=torch.float32)
    M = torch.tensor(d[mask_key], dtype=torch.float32)
    _cap = 200_000
    if X.shape[0] > _cap:
        X, M = X[:_cap], M[:_cap]
        print(f"subsampled to {_cap} for speed")
    B, max_n, dim = X.shape
    print(f"tokens {tuple(X.shape)}  mask {tuple(M.shape)}  hidden(fixed)={a.hidden}\n")

    results = {}
    for out in a.sweep_output:
        print(f"########## output width = {out} ##########")
        print(f"--- DeepSetsAE ---")
        ds = train(DeepSetsAE(dim, a.hidden, out, max_n), X, M, a.epochs, a.batch_size)
        print(f"--- SetAttnAE ---")
        at = train(SetAttnAE(dim, a.hidden, out, max_n), X, M, a.epochs, a.batch_size)
        results[out] = (ds, at)

    print("\n================ SUMMARY (lower=better) ================")
    print(f"{'output':>8} {'DeepSets':>12} {'SetAttn':>12}")
    for out,(ds,at) in results.items():
        print(f"{out:>8} {ds:>12.4f} {at:>12.4f}")
