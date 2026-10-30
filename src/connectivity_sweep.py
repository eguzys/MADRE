"""Drone-backbone connectivity: P(two random drones connected). Multiprocess."""
import argparse, itertools, time, os
import numpy as np
import networkx as nx
from multiprocessing import Pool

ARENA = 100.0

def rgg_null_pair(radius, arena=ARENA, trials=200_000, seed=0):
    """P(two uniform-iid points within radius) — the direct-pair null."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, arena, size=(trials, 2, 2))
    d = np.hypot(p[:,0,0]-p[:,1,0], p[:,0,1]-p[:,1,1])
    return np.mean(d <= radius)

_CFG = {}

def _init_worker(policy_name, n_drones, n_users, power, max_steps, state_loss):
    from MANET import Environments as E
    from MANET.Components import MANETNode
    from MANET.build import build_padre_env
    from MANET.Parallel_enviroments import PADREParallelEnv
    from evaluation.heuristics import GreedyHeuristic
    E.Static_MANETEnv.nodeSignalPow_mW = power

    if policy_name == "greedy":
        policy = GreedyHeuristic("sinr")
    elif policy_name == "random":
        class _Rand:
            def reset(self, env, seed=None): self._rng = np.random.default_rng(seed)
            def __call__(self, obs, na):
                return {f"drone_{i}": self._rng.uniform(-1,1,size=2).astype(np.float32)
                        for i in range(na)}
        policy = _Rand()
    else:
        raise ValueError(policy_name)

    env = PADREParallelEnv(build_env_fn=lambda: build_padre_env(
        numbOfNodes=n_drones, numbOfJammers=1, numbOfUsers=n_users,
        state_loss_rate=state_loss, render_mode=None))
    _CFG.update(env=env, policy=policy, n_drones=n_drones,
                max_steps=max_steps, MANETNode=MANETNode)

def _run_seed(seed):
    env = _CFG["env"]; policy = _CFG["policy"]
    n_drones = _CFG["n_drones"]; max_steps = _CFG["max_steps"]
    MANETNode = _CFG["MANETNode"]

    obs, _ = env.reset(seed=seed)
    na = len(env.possible_agents)
    if hasattr(policy, "reset"): policy.reset(env, seed)
    direct_frac, reach_frac, comp_list, deg_list = [], [], [], []
    step_idx, done = 0, False
    while not done:
        obs, r, t, tr, _ = env.step(policy(obs, na))
        G = env._wrapper.unwrapped_env.network.getGraph()
        drones = [n for n in G.nodes if isinstance(n, MANETNode)]
        UG = G.subgraph(drones).to_undirected()
        nodes = list(UG.nodes())
        m = len(nodes)
        pairs = list(itertools.combinations(range(m), 2))
        npairs = len(pairs) if pairs else 1

        direct = sum(1 for a, b in pairs if UG.has_edge(nodes[a], nodes[b]))

        comp_of = {}
        for ci, cc in enumerate(nx.connected_components(UG)):
            for node in cc:
                comp_of[node] = ci
        reach = sum(1 for a, b in pairs if comp_of.get(nodes[a]) == comp_of.get(nodes[b]))

        direct_frac.append(direct / npairs)
        reach_frac.append(reach / npairs)
        comp_list.append(nx.number_connected_components(UG))
        degs = [d for _, d in UG.degree()]
        deg_list.append(np.mean(degs) if degs else 0.0)

        step_idx += 1
        done = any(t.values()) or any(tr.values()) or (len(env.agents) == 0) \
               or (max_steps is not None and step_idx >= max_steps)
    return (np.mean(direct_frac), np.mean(reach_frac),
            np.mean(comp_list), np.mean(deg_list))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["greedy","random"], default="greedy")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--drones", type=int, default=4)
    ap.add_argument("--users", type=int, default=15)
    ap.add_argument("--power", type=float, default=200)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count()-1))
    ap.add_argument("--state-loss", type=float, default=0.3)
    args = ap.parse_args()

    from MANET.Components import MANETNode
    radius = MANETNode("p", np.array([0.0,0.0]), signalPower_mW=args.power).approxRadius()
    null_direct = rgg_null_pair(radius)
    print(f"power={args.power}mW radius={radius:.2f} arena={ARENA:.0f} "
          f"radius/diag={radius/(ARENA*np.sqrt(2)):.3f}", flush=True)
    print(f"RGG null: P(two uniform drones within range) = {null_direct:.3f}\n", flush=True)
    print(f"policy={args.policy} seeds={args.seeds} workers={args.workers} "
          f"max_steps={args.max_steps} state_loss={args.state_loss}", flush=True)

    t0 = time.time()
    results = []
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(args.policy, args.drones, args.users, args.power, args.max_steps, args.state_loss)) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_seed, range(args.seeds)), 1):
            results.append(res)
            el = time.time()-t0
            print(f"  {i:>3}/{args.seeds} | {el:.0f}s | ETA {el/i*(args.seeds-i)/60:.1f} min",
                  flush=True)

    arr = np.array(results)
    names = ["pair_direct", "pair_reachable", "mean_components", "mean_degree"]
    print(f"\n{'metric':>18} {'mean':>8} {'std':>8}\n{'-'*38}")
    for j, nm in enumerate(names):
        print(f"{nm:>18} {arr[:,j].mean():>8.3f} {arr[:,j].std():>8.3f}")
    pd = arr[:,0].mean()
    print(f"\n  pair_direct {pd:.3f} vs RGG null {null_direct:.3f}: "
          f"{'ABOVE' if pd>null_direct else 'BELOW'} by {abs(pd-null_direct):.3f}")
