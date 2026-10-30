"""widen2.py -- widen a 40-user OM2G checkpoint to a new user cap.

Pads the critic weight and every cap-dependent buffer tensor (state AND
per-agent observation) so a 100-user fine-tune can resume from a 40-user run.

Cap-dependent tensors in a checkpoint (verified by dumping every tensor of
u100diag_ckpts/L128_sp4_s1/checkpoints/checkpoint_216000.pt):
  loss_drone/critic_network_params.module.0.mlp.0.0.weight   (256, S)
  buffer_drone/_storage/_storage/state                        (6000, S)
  buffer_drone/_storage/_storage/next/state                   (6000, S)
  buffer_drone/_storage/_storage/drone/observation            (6000, 8, O)
  buffer_drone/_storage/_storage/next/drone/observation       (6000, 8, O)
with S = 4*MAX_N + 4*cap + 2 (omnet_module/omnet_env.py _build_state:
[0:32) drone blocks | [32:32+4cap) user blocks | steps_remaining | frame_valid),
so the NEW user columns are INSERTED at 32+4*cap_old, keeping the two
trailing columns at the end. The rollout buffer is one
6000-frame on-policy batch that PPO empties and refills every iteration, so
its CONTENT is irrelevant -- only the shapes must match the new env.

O (obs width) is taken from the caller (--obs-new), because it depends on the
arm (talker 7304 -> 16484 at cap 100; nocontrol 840 -> 1860); the value must
come from the env's own banner (`[OM2GEnv] ... obs=16484`), never guessed.
The script also asserts O_new - O_old == (cap_new-cap_old)*17*(1+n_peers)
with n_peers in {0, 8} as a cross-check.

Usage:
  widen2.py <src_ckpt.pt> <cap_new> <obs_new> <dst_run_dir>
Writes <dst_run_dir>/checkpoints/<same name>, copies config.pkl + *.json from
the source experiment dir (src_ckpt/../..), then RELOADS the written file and
verifies: every non-widened tensor bit-identical; widened tensors have the
expected shapes; critic columns mapped as specified. Exit 1 on any failure.
"""
import glob
import os
import shutil
import sys

import torch

MAX_N = 8
OBS_FEATURES = 17
DRONE_COLS = 4 * MAX_N          # 32
TRAIL = 2                       # steps_remaining_norm, frame_valid

K_CRITIC = "critic_network_params.module.0.mlp.0.0.weight"


def walk(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from walk(v, p + "/" + str(k))
    elif isinstance(d, torch.Tensor):
        yield p, d


def widen_state_cols(t, cap_old, cap_new):
    """Insert 4*(cap_new-cap_old) zero columns at DRONE_COLS + 4*cap_old (last dim)."""
    S_old = DRONE_COLS + 4 * cap_old + TRAIL
    S_new = DRONE_COLS + 4 * cap_new + TRAIL
    assert t.shape[-1] == S_old, f"state width {t.shape[-1]} != {S_old}"
    cut = DRONE_COLS + 4 * cap_old
    shape = list(t.shape)
    shape[-1] = S_new
    out = torch.zeros(shape, dtype=t.dtype)
    out[..., :cut] = t[..., :cut]
    out[..., S_new - TRAIL:] = t[..., S_old - TRAIL:]
    return out


def main():
    src, cap_new, dst_run = sys.argv[1], int(sys.argv[2]), sys.argv[4]
    obs_arg = sys.argv[3]  # int, or "auto" = obs_old + (cap_new-cap_old)*17*9 (self + 8 peer inbox), guarded by the resume's own shape check
    src = os.path.abspath(src)
    src_exp = os.path.dirname(os.path.dirname(src))
    name = os.path.basename(src)
    print(f"[widen2] loading {src}", flush=True)
    A = torch.load(src, map_location="cpu", weights_only=False)

    W_crit = A["loss_drone"][K_CRITIC]
    cap_old = (W_crit.shape[1] - DRONE_COLS - TRAIL) // 4
    assert DRONE_COLS + 4 * cap_old + TRAIL == W_crit.shape[1], f"critic in-width {W_crit.shape[1]} not 34+4*cap"
    assert cap_new > cap_old, f"cannot shrink cap {cap_old} -> {cap_new}"

    st = A["buffer_drone"]["_storage"]["_storage"]
    if st is None:
        raise SystemExit("buffer storage is None: buffer-stripped checkpoint cannot resume")
    obs_old = st["drone"]["observation"].shape[-1]
    obs_new = obs_old + (cap_new - cap_old) * OBS_FEATURES * (1 + MAX_N) if obs_arg == "auto" else int(obs_arg)
    d_obs = obs_new - obs_old
    per_cap = d_obs / (cap_new - cap_old)
    n_peers = per_cap / OBS_FEATURES - 1
    assert per_cap == int(per_cap) and n_peers in (0, MAX_N), (
        f"obs_new {obs_new} - obs_old {obs_old} = {d_obs} is not (cap_new-cap_old)*17*(1+n_peers): "
        f"per_cap={per_cap}, n_peers={n_peers}")
    print(f"[widen2] cap {cap_old}->{cap_new}  state {W_crit.shape[1]}->{DRONE_COLS + 4*cap_new + TRAIL}  "
          f"obs {obs_old}->{obs_new} (n_peers={int(n_peers)})", flush=True)

    # --- record originals for verification ---
    orig = {k: v.clone() for k, v in walk(A)}

    # --- widen the five tensors ---
    A["loss_drone"][K_CRITIC] = widen_state_cols(W_crit, cap_old, cap_new)
    st["state"] = widen_state_cols(st["state"], cap_old, cap_new)
    st["next"]["state"] = widen_state_cols(st["next"]["state"], cap_old, cap_new)
    for holder in (st["drone"], st["next"]["drone"]):
        o = holder["observation"]
        z = torch.zeros((o.shape[0], o.shape[1], obs_new), dtype=o.dtype)
        # content is discarded by PPO before any read; keep old columns anyway
        z[..., :obs_old] = o
        holder["observation"] = z
    widened = {
        "/loss_drone/" + K_CRITIC,
        "/buffer_drone/_storage/_storage/state",
        "/buffer_drone/_storage/_storage/next/state",
        "/buffer_drone/_storage/_storage/drone/observation",
        "/buffer_drone/_storage/_storage/next/drone/observation",
    }

    # any OTHER tensor with a cap-dependent dimension is not handled by this script
    S_old = DRONE_COLS + 4 * cap_old + TRAIL
    suspicious = [k for k, v in walk(A) if k not in widened and v.dim() >= 1
                  and any(int(s) in (S_old, obs_old) for s in v.shape) and int(v.numel()) > 1]
    if suspicious:
        raise SystemExit(f"[widen2] UNHANDLED cap-dependent tensors: {suspicious}")

    os.makedirs(os.path.join(dst_run, "checkpoints"), exist_ok=True)
    for f in glob.glob(os.path.join(src_exp, "*.json")) + [os.path.join(src_exp, "config.pkl")]:
        if os.path.exists(f):
            shutil.copy2(f, dst_run)
    assert os.path.exists(os.path.join(dst_run, "config.pkl")), "config.pkl missing in staging"
    out = os.path.join(dst_run, "checkpoints", name)
    tmp = out + ".partial"
    torch.save(A, tmp)
    del A

    # --- verify by reloading ---
    B = torch.load(tmp, map_location="cpu", weights_only=False)
    got = dict(walk(B))
    bad = []
    for k, v in orig.items():
        if k in widened:
            continue
        if k not in got or got[k].shape != v.shape or not torch.equal(got[k], v):
            bad.append(k)
    for k in widened:
        if k not in got:
            bad.append("MISSING " + k)
    S_new = DRONE_COLS + 4 * cap_new + TRAIL
    cut = DRONE_COLS + 4 * cap_old
    wc, wo = got["/loss_drone/" + K_CRITIC], orig["/loss_drone/" + K_CRITIC]
    ok_crit = (wc.shape == (wo.shape[0], S_new)
               and torch.equal(wc[:, :cut], wo[:, :cut])
               and torch.equal(wc[:, S_new - TRAIL:], wo[:, S_old - TRAIL:])
               and bool((wc[:, cut:S_new - TRAIL] == 0).all()))
    if not ok_crit:
        bad.append("critic column mapping")
    for k in ("/buffer_drone/_storage/_storage/state", "/buffer_drone/_storage/_storage/next/state"):
        if got[k].shape != (orig[k].shape[0], S_new):
            bad.append(k + " shape")
    for k in ("/buffer_drone/_storage/_storage/drone/observation",
              "/buffer_drone/_storage/_storage/next/drone/observation"):
        if got[k].shape != (orig[k].shape[0], orig[k].shape[1], obs_new):
            bad.append(k + " shape")
    n_total = len(got)
    if bad:
        os.remove(tmp)
        raise SystemExit(f"[widen2] VERIFY FAILED ({len(bad)}): {bad[:8]}")
    os.replace(tmp, out)
    print(f"[widen2] OK {out} ({os.path.getsize(out)/1e9:.2f} GB): {n_total} tensors, "
          f"{n_total - len(widened)} bit-identical, 5 widened; critic cols [0:{cut})->[0:{cut}), "
          f"[{S_old-TRAIL}:{S_old})->[{S_new-TRAIL}:{S_new}), new [{cut}:{S_new-TRAIL}) zero", flush=True)


if __name__ == "__main__":
    main()
