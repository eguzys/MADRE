#!/usr/bin/env python3
"""Behaviour-clone relay_blind onto the compressed-arm st_model actor.

WHY relay_blind AND NOT relay_informed. The oracle sees all 40 users; the
student sees own-radio + the 88 B message. You cannot clone what the student
cannot perceive -- and a linear probe on the oracle dump recovered nothing
(R2 ~ 0). relay_blind uses POSITIONS ONLY, which is far closer to the student's
actual observability, and it IS the C2 gate (123.2).

WHAT THE CLONE IS FOR. Behaviour cloning only sets the initial weights of the
actor; the policy is then trained and evaluated on the true unshaped reward
under matched charging, as with curriculum and exploring starts. The test is
(i) whether the final policy exceeds 123.2 at n=100 while paying 131 B against
relay_blind's 51 B, AND (ii) whether the payload ablation brings it back toward
relay_blind, which shows the margin depends on message CONTENT rather than on
geometry.

DESIGN CHOICES, so they are not re-derived later:
* Huber on the PRE-SQUASH loc, not Gaussian NLL. Evaluation is deterministic
  (tanh(loc)), so loc is the deployed quantity; NLL would drag the scale head
  into the fit and sigma must survive as the fine-tune's exploration budget.
  loc rows and scale rows of _output_proj are disjoint, so a loc-only loss
  leaves sigma untouched by construction.
* atanh with a 0.997 clamp: relay_blind's _fly_to saturates at +-1 constantly,
  and tanh-output MSE has vanishing gradient exactly there.
* The student IS the real actor, rebuilt by the same loader evaluation uses --
  not a fresh net. That rules out constructor drift, and
  its state_dict keys are the checkpoint's actor_network_params.* suffixes by
  construction.
* Split by EPISODE, never by sample: consecutive steps are highly correlated
  and a random split would leak.

GO/NO-GO on validation cosine: >=0.6 proceed | 0.3-0.6 partial clone, the eval
reveals the cap | <0.3 STOP and report that the observability cap is binding --
that would mean no shaping redesign could make PPO express relay behaviour
either, and curriculum becomes the only track.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential

sys.path.insert(0, os.getcwd())                       # must run from src/
from evaluation.policies import _load_via_reload_from_file  # noqa: E402

DONOR, PAIRS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

policy, _task_cfg, _fam = _load_via_reload_from_file(Path(DONOR), "cpu")
policy.train()
param_net = TensorDictSequential(*list(policy.module)[:-1])   # drops the
#                                   ProbabilisticTensorDictModule -> loc/scale

# PAIRS may be a single json OR a DIRECTORY of chunk dumps. The dump is
# chunked across the seed band so it finishes in ~25 min instead of ~4 h, and
# the chunks are parsed ONE AT A TIME: the merged json is >1 GB and json.load
# on the whole thing peaks at ~10x that in RAM, while the extracted float32
# arrays are only ~0.7 GB.
_paths = (sorted(Path(PAIRS).glob("*.json")) if Path(PAIRS).is_dir()
          else [Path(PAIRS)])
if not _paths:
    sys.exit(f"no dump files at {PAIRS}")
samples = []
for _p in _paths:
    by_key = {}
    for r in json.load(open(_p)):
        if r.get("kind") != "obs":
            continue
        by_key.setdefault((r["seed"], r["step"]), {})[r["agent"]] = r
    for (seed, _step), ag in sorted(by_key.items()):
        # the dumper writes act=None when an agent was absent from the action
        # dict that step; a partial group would silently mis-pair obs to act.
        if len(ag) != 8 or any(ag.get(f"drone_{i}") is None
                               or ag[f"drone_{i}"].get("act") is None
                               for i in range(8)):
            continue
        samples.append((
            seed,
            np.stack([ag[f"drone_{i}"]["obs"] for i in range(8)]).astype(np.float32),
            np.stack([ag[f"drone_{i}"]["act"] for i in range(8)]).astype(np.float32),
        ))
    del by_key
    print(f"  {_p.name}: {len(samples)} cumulative samples", flush=True)
seeds = sorted({s for s, _, _ in samples})
val_seeds = set(seeds[::10])
tr = [(o, a) for s, o, a in samples if s not in val_seeds]
va = [(o, a) for s, o, a in samples if s in val_seeds]
print(f"  samples train={len(tr)} val={len(va)} episodes={len(seeds)}", flush=True)
if not tr or not va:
    sys.exit("empty split -- check the dump")


def forward_loc(obs_b):
    td = TensorDict({"drone": TensorDict({"observation": obs_b},
                                         batch_size=[obs_b.shape[0]])},
                    batch_size=[obs_b.shape[0]])
    out = param_net(td)["drone"]
    for k in ("loc", "param_loc"):
        if k in out.keys():
            return out[k]
    raise KeyError(f"no loc key; got {list(out.keys())}")


def pre_squash(a):
    return torch.atanh(torch.clamp(a, -0.997, 0.997))


opt = torch.optim.Adam([p for p in param_net.parameters() if p.requires_grad],
                       lr=3e-4)                       # weight_decay MUST be 0
huber = torch.nn.HuberLoss(delta=1.0)
B = 256


def run_epoch(data, train):
    idx = np.random.permutation(len(data)) if train else np.arange(len(data))
    tot, cos_sum, n = 0.0, 0.0, 0
    for s in range(0, len(idx), B):
        chunk = [data[i] for i in idx[s:s + B]]
        obs = torch.from_numpy(np.stack([c[0] for c in chunk]))
        act = torch.from_numpy(np.stack([c[1] for c in chunk]))
        with torch.set_grad_enabled(train):
            loc = forward_loc(obs)
            loss = huber(loc, pre_squash(act))
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(param_net.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            cs = torch.nn.functional.cosine_similarity(
                torch.tanh(loc).flatten(0, 1), act.flatten(0, 1), dim=-1)
            cos_sum += float(cs.sum())
            n += cs.numel()
        tot += float(loss) * len(chunk)
    return tot / len(idx), cos_sum / max(n, 1)


best, patience = 1e9, 0
for ep in range(200):
    trl, trc = run_epoch(tr, True)
    with torch.no_grad():
        val, vac = run_epoch(va, False)
    print(f"  ep {ep:3d} train {trl:.4f}/cos {trc:.3f}   val {val:.4f}/cos {vac:.3f}",
          flush=True)
    if val < best - 1e-4:
        best, patience = val, 0
        torch.save(policy.state_dict(), OUT)
    else:
        patience += 1
        if patience >= 8:
            break
print(f"  best val {best:.4f} -> {OUT}", flush=True)
