"""Null the replay buffer in a widened checkpoint so a resume can rebuild it.

The buffer carries the per-agent observation, which scales at 153 floats/user;
keeping a 40-user buffer in a 100-user run would not match the new shapes.

BenchMARL experiment.load_state_dict skips the buffer when buffer_<group> is
None (experiment.py:987), and MAPPO is on-policy: extend() overwrites all 6000
slots before anything is sampled. So dropping it loses nothing and the file
shrinks from GBs to MBs.

Usage: dropbuf.py <ckpt.pt> [...]   -- rewrites in place, atomically.
"""
import os, sys, torch

for path in sys.argv[1:]:
    A = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    buf = A.get("buffer_drone")
    if buf is None:
        print(f"[dropbuf] SKIP already-null  {path}", flush=True); continue
    st = buf["_storage"]["_storage"]
    obs = tuple(st["drone"]["observation"].shape)[-1]
    sw  = tuple(st["state"].shape)[-1]
    cap = (sw - 34) // 4
    before = os.path.getsize(path)
    A["buffer_drone"] = None
    tmp = path + ".dropbuf.tmp"
    torch.save(A, tmp)
    os.replace(tmp, path)
    after = os.path.getsize(path)
    print(f"[dropbuf] state={sw} (cap {cap}) obs={obs} dropped  "
          f"{before/1e9:.2f}GB -> {after/1e6:.1f}MB  {path}", flush=True)
