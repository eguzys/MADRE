"""Merge per-seed OMNeT collections into one episode-ordered npz.

CRITICAL -- the flatten:
  collected: user_tokens (T, 5, 25, 13)   -- 5 DRONES per timestep
  encoder wants: (N, 25, 13)              -- one set per row
So we flatten (T,5) -> T*5. Each drone's sensed set is an independent set.

CRITICAL -- the split:
  episode_id must be REPEATED 5x (once per drone) so the episode-aligned split
  stays aligned. A random-row split leaks: step 437 and 438 are near-identical,
  and the SAME drone at t and t+1 is even more so.
Files are concatenated in SEED ORDER so the trailing slice is whole episodes.
"""
import argparse, glob, os, re
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--glob", default="/data/omnet_sets_seed*.npz")
p.add_argument("--out",  default="/data/omnet_sets_merged.npz")
a = p.parse_args()

files = sorted(glob.glob(a.glob),
               key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
print(f"{len(files)} files, seeds {files[0]} .. {files[-1]}")

UT, ET, UM, EM, EP = [], [], [], [], []
for f in files:
    d = np.load(f)
    T = d["user_tokens"].shape[0]
    UT.append(d["user_tokens"].reshape(T * 5, 25, 13))
    ET.append(d["event_tokens"].reshape(T * 5, 32, 12))
    UM.append(d["user_mask"].reshape(T * 5, 25))
    EM.append(d["event_mask"].reshape(T * 5, 32))
    EP.append(np.repeat(d["episode_id"], 5))          # 5 drones share an episode

tokens = np.concatenate(UT).astype(np.float32)
masks  = np.concatenate(UM).astype(np.float32)
ev_tok = np.concatenate(ET).astype(np.float32)
ev_msk = np.concatenate(EM).astype(np.float32)
epid   = np.concatenate(EP).astype(np.int64)

# `tokens`/`masks` keys match what train_final_encoder.py / distributional_recon
# expect (it sniffs for a key containing "mask").
np.savez_compressed(a.out, tokens=tokens, masks=masks,
                    event_tokens=ev_tok, event_masks=ev_msk, episode_id=epid)

s = masks.sum(-1)
print(f"tokens {tokens.shape}  episodes {len(np.unique(epid))}")
print(f"rows/episode {len(epid)//len(np.unique(epid))}  (expect 5000 = 1000 steps x 5 drones)")
print(f"set size: mean {s.mean():.2f}  p50 {np.percentile(s,50):.0f}  "
      f"max {s.max():.0f}  frac_empty {(s==0).mean():.3f}")
print(f"wrote {a.out}  {os.path.getsize(a.out)/1e6:.1f} MB")
