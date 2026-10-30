"""Train PISA (permutation-invariant set autoencoder) on collected drone
user-sets, sweeping over hidden_dim.

Mirrors the training loop in sae/run/train_rand.py, but consumes real
collected data (from collect_user_sets.py) instead of random sets, and
converts PADRE's padded+masked rectangular format into PISA's sparse
(x, batch) format.

PISA sparse format:
  x     : (total_real, dim) -- all real rows from every sample in the chunk,
          stacked in sample order.
  batch : (total_real,)     -- for each row in x, which sample (0..n-1
          within the chunk) it belongs to.
  n_batches : int            -- number of samples in the chunk. Must be
          passed explicitly to the model: if a sample has 0 real rows (an
          empty set, e.g. a drone that senses no one), it contributes no
          entries to `batch`, so inferring n_batches from batch.max()+1
          would silently drop trailing empty samples.

Usage:
  python train_pisa_sweep.py --npz user_sets.npz --epochs 5000 \
      --batch-size 64 --out-dir pisa_out --hidden-dims 32 64 96 128
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.optim import Adam

from sae import AutoEncoder


def sparse_from_padded(tokens, masks, start, end):
    """Convert a padded+masked (chunk, W, dim) slice to PISA's sparse format.

    tokens: (M, W, dim) float array, masks: (M, W) 0/1 array. Only rows
    where mask==1 (real, sensed slots) become part of x; padded/unsensed
    slots are dropped.
    """
    tok_chunk = tokens[start:end]  # (n, W, dim)
    mask_chunk = masks[start:end]  # (n, W)
    n_batches = tok_chunk.shape[0]

    mask_bool = mask_chunk.astype(bool)
    x = tok_chunk[mask_bool]  # (total_real, dim), row-major order
    row_idx, _ = np.nonzero(mask_bool)  # sample index per real row, same order
    batch = row_idx.astype(np.int64)

    assert batch.size == int(mask_chunk.sum())
    assert n_batches == tok_chunk.shape[0]
    if batch.size > 0:
        assert batch.max() < n_batches

    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    batch = torch.from_numpy(batch)
    return x, batch, n_batches


def iterate_batches(tokens, masks, batch_size, shuffle=True):
    """Yield (x, batch, n_batches) chunks covering the whole dataset once."""
    M = tokens.shape[0]
    order = np.random.permutation(M) if shuffle else np.arange(M)
    tokens_shuf = tokens[order]
    masks_shuf = masks[order]
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        yield sparse_from_padded(tokens_shuf, masks_shuf, start, end)


def forward_loss(model, x, batch, n_batches, device):
    """Run one PISA forward+loss pass, passing n_batches explicitly."""
    x = x.to(device)
    batch = batch.to(device)
    z = model.encoder(x, batch, n_batches=n_batches)
    xr, batchr = model.decoder(z)
    loss_data = model.loss()
    return loss_data, xr, batchr


def train_hidden_dim(tokens, masks, hidden_dim, num_steps, batch_size, device,
                      print_at=()):
    dim = tokens.shape[2]
    max_n = tokens.shape[1]
    model = AutoEncoder(dim=dim, hidden_dim=hidden_dim, max_n=max_n).to(device)
    optim = Adam(model.parameters())

    curve = []
    step = 0
    xr = batchr = None
    while step < num_steps:
        for x, batch, n_batches in iterate_batches(tokens, masks, batch_size):
            if step >= num_steps:
                break
            loss_data, xr, batchr = forward_loss(model, x, batch, n_batches, device)
            loss = loss_data["loss"]

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_val = float(loss.item())
            curve.append(loss_val)
            if step in print_at:
                print(f"    step {step:4d} loss={loss_val:.4f}")
            step += 1
    return model, curve, xr, batchr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5000,
                     help="number of gradient steps per hidden_dim")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", type=str, default="pisa_sweep_out")
    ap.add_argument("--hidden-dims", type=int, nargs="+",
                     default=[32, 64, 96, 128])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.npz)
    tokens = data["tokens"]
    masks = data["masks"]
    print(f"loaded {args.npz}: tokens {tokens.shape}, masks {masks.shape}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for h in args.hidden_dims:
        print(f"=== training hidden_dim={h} ===")
        model, curve, _, _ = train_hidden_dim(
            tokens, masks, h, args.epochs, args.batch_size, device
        )
        state_path = os.path.join(args.out_dir, f"pisa_h{h}.pt")
        torch.save(model.state_dict(), state_path)
        log_path = os.path.join(args.out_dir, f"pisa_h{h}_loss.json")
        with open(log_path, "w") as f:
            json.dump({"final_loss": curve[-1], "curve": curve}, f)
        print(f"  final loss={curve[-1]:.4f} -> saved {state_path}, {log_path}")


if __name__ == "__main__":
    main()
