"""Diagnostic wrapper around evaluate_omnet.main(): records the statistics of every
LearnableUserSetEncoder output (self path out_dim=64, codec E out_dim=rt_width) so the
message latent at 100 users can be compared with 40 users. Writes $HOOK_OUT (JSON)."""
import os, sys, json, atexit
import numpy as np
OM = "/usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone/omnet_module"
sys.path.insert(0, OM)
import evaluate_omnet as EV  # noqa  (adds src to sys.path)
import torch
from MANET import frozen_encoder as FE

HOOK_OUT = os.environ["HOOK_OUT"]
RS = 3000
rng = np.random.default_rng(0)
acc = {}

def rec(key, out, mask):
    with torch.no_grad():
        o = out.detach().reshape(-1, out.shape[-1]).float()
        nl = mask.detach().reshape(-1, mask.shape[-1]).float().sum(-1)
        live = nl > 0
        o = o[live]; nl = nl[live]
        if o.shape[0] == 0:
            return
        on = o.cpu().numpy(); nln = nl.cpu().numpy()
        a = acc.get(key)
        if a is None:
            a = acc[key] = dict(n=0, s=np.zeros(on.shape[1]), ss=np.zeros(on.shape[1]),
                                mn=np.full(on.shape[1], np.inf), mx=np.full(on.shape[1], -np.inf),
                                sample=[], nlive=[], nl_sum=0.0, nl_max=0.0, calls=0)
        a["calls"] += 1
        a["s"] += on.sum(0); a["ss"] += (on ** 2).sum(0)
        a["mn"] = np.minimum(a["mn"], on.min(0)); a["mx"] = np.maximum(a["mx"], on.max(0))
        a["nl_sum"] += float(nln.sum()); a["nl_max"] = max(a["nl_max"], float(nln.max()))
        for i in range(on.shape[0]):
            a["n"] += 1
            if len(a["sample"]) < RS:
                a["sample"].append(on[i].tolist()); a["nlive"].append(float(nln[i]))
            else:
                j = int(rng.integers(0, a["n"]))
                if j < RS:
                    a["sample"][j] = on[i].tolist(); a["nlive"][j] = float(nln[i])

_orig = FE.LearnableUserSetEncoder.forward
def _patched(self, tokens, mask):
    out = _orig(self, tokens, mask)
    try:
        rec(f"setenc_out{self.out_dim}", out, mask)
    except Exception as e:  # never take the eval down
        print(f"[hook] rec failed: {e}", flush=True)
    return out
FE.LearnableUserSetEncoder.forward = _patched
print(f"[hook] LearnableUserSetEncoder.forward patched; HOOK_OUT={HOOK_OUT}", flush=True)

# Per-step sender->partner map: count how many pairs change between consecutive steps, so a
# demand reshuffle (OM2G_DEMAND_RESHUFFLE_STEPS) is verified from the simulator side.
PAIRS = dict(steps=0, changes=0, changed_steps=0, prev=None)
_orig_sel = EV._select_actions
def _sel(policy, action_fn, obs_dict, num_drones, env):
    try:
        lo = getattr(env, "_last_obs", None) or {}
        cur = tuple(tuple(int(v) for v in p) for p in lo.get("sender_pairs", []))
        PAIRS["steps"] += 1
        if PAIRS["prev"] is not None and len(cur) == len(PAIRS["prev"]) and len(cur) > 0:
            nchg = sum(1 for a, b in zip(cur, PAIRS["prev"]) if a != b)
            PAIRS["changes"] += nchg
            PAIRS["changed_steps"] += int(nchg > 0)
        PAIRS["prev"] = cur
    except Exception as e:
        print(f"[hook] pairs rec failed: {e}", flush=True)
    return _orig_sel(policy, action_fn, obs_dict, num_drones, env)
EV._select_actions = _sel
print("[hook] evaluate_omnet._select_actions patched for pair-change counting", flush=True)

def _dump():
    out = {"pairs": {k: v for k, v in PAIRS.items() if k != "prev"}}
    print(f"[hook] pairs: {out['pairs']}", flush=True)
    for k, a in acc.items():
        n = max(a["n"], 1)
        mean = a["s"] / n
        var = np.maximum(a["ss"] / n - mean ** 2, 0.0)
        out[k] = dict(n=a["n"], calls=a["calls"], mean=mean.tolist(), std=np.sqrt(var).tolist(),
                      min=a["mn"].tolist(), max=a["mx"].tolist(),
                      nlive_mean=a["nl_sum"] / n, nlive_max=a["nl_max"],
                      sample=a["sample"], nlive=a["nlive"])
    with open(HOOK_OUT, "w") as fh:
        json.dump(out, fh)
    print("[hook] wrote " + HOOK_OUT + ": " + ", ".join(
        f"{k} n={v['n']} nlive_mean={v['nlive_mean']:.2f}" for k, v in out.items()), flush=True)
atexit.register(_dump)

sys.argv = ["evaluate_omnet.py"] + sys.argv[1:]
EV.main()
