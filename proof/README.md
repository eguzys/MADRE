# MadreProof: machine-checked proofs for the MADRE paper and its theory note

This repository holds Lean 4 + Mathlib proofs of Theorem 1 of the MADRE paper (drone relays
that exchange learned messages of size L) and of the results of the accompanying theory note
that the paper's text relies on. Every statement below is fully proved. There is no `sorry`,
`admit`, `native_decide` or new `axiom`. `lake build` prints the axioms each main theorem uses,
and they are only Lean's three standard ones, `[propext, Classical.choice, Quot.sound]`.

## How to build

1. Install `elan`, the Lean version manager:
   `curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh`
   (it reads the Lean version from `lean-toolchain`).
2. In this directory, download the compiled Mathlib (about 5 GB) instead of building it:
   `lake exe cache get`
3. Build and check everything: `lake build`

The last lines of the output are the `#print axioms` lines of `MadreProof/Check.lean`. GitHub
Actions runs the same steps on every push (`.github/workflows/lean.yml`) and fails if `sorryAx` or
any non-standard axiom appears. The warning "imports `Mathlib` ... designed for use with the
module system" is harmless.

## What is machine-checked, and what is not

| paper or note | Lean statement | file |
|---|---|---|
| Theorem 1(a), the airtime bound (eq. air) | `PropositionA` | `Statement.lean` |
| "λ_t(i) is the multiplier of an optimal dual": an optimal dual exists and has value τ⁰ (LP strong duality) | `LPStrongDuality` | `Statement.lean` |
| "τ_t, τ⁰_t the optimal values of (P1)": the optimum exists (c, D ≥ 0, κ(i) ≤ R) | `LPOptimumExists` | `Statement.lean` |
| "λ_t(i) vanishes while drone i has airtime to spare" (complementary slackness, this direction only) | `ComplementarySlackness` | `Statement.lean` |
| Theorem 1(b), the information bound | `PropositionB` | `Statement.lean` |
| Theorem 1(b), reading "one ω common to all M observations": both conclusions at ω₀, and "answered well at every such ω" implies "at ω₀" | `PropositionBCommonOmega` | `Statement.lean` |
| Note, Lemma lem:pdl (performance difference) | `PerformanceDifference` | `Statement/Throughput.lean` |
| Note, Theorem thm:floor (ii) | `TheoremFloorII` | `Statement/Throughput.lean` |
| Note, Theorem thm:floor (iii): "each miss costing throughput as often as the trajectory meets that observation" (finite setting, discrete-move or deterministic receivers, see below) | `TheoremFloorIII`, `ReceiverFiniteSupport` | `Statement/Throughput.lean` |
| Note, Lemma lem:pdl and Theorem thm:floor (iii) on a GENERAL probability space (any receiver law, Gaussian and tanh-Normal included) | `PerformanceDifferenceGeneral`, `TheoremFloorIIIGeneral` | `Statement/ThroughputGeneral.lean` |
| The Markov hypothesis of the general version follows from a transition kernel | `MarkovOfKernel` | `Statement/ThroughputGeneral.lean` |
| (P1) with an explicit edge set E (as the paper writes it): same feasible values as the all-pairs LP with c = 0 off E, and Theorem 1(a) for it | `EdgeSetValuesEq`, `PropositionAEdgeSet` | `Statement/EdgeSet.lean` |
| (P1) with a WEIGHTED radio row Σ a_i(v,w) Σ_k f_k(v,w) ≤ R − κ(i) (the scorer's LP, a = 1/p on transmit edges): strong duality, optimum existence, Theorem 1(a), complementary slackness, witness scaling, charge antitone, the two-sided bracket at the simulator constants (same floor factors), the explicit-edge version, a tight non-vacuity instance | `LPStrongDualityW`, `LPOptimumExistsW`, `PropositionAW`, `ComplementarySlacknessW`, `WitnessScalingW`, `ChargeAntitoneW`, `AirtimeBracketW`, `EdgeSetValuesEqW`, `PropositionAEdgeSetW`, `propositionAW_nonvacuous` | `Statement/Weighted.lean` |
| The paper's unit-weight (P1) is the special case; the unit theorems re-derived from the weighted ones | `UnitWeightSpecialCase`, `propositionA_fromWeighted`, `lpStrongDuality_fromWeighted`, `complementarySlackness_fromWeighted`, `airtimeBracket_fromWeighted` | `Statement/Weighted.lean` |
| The REALIZED move (torchrl safe-tanh clamp, wall reflection) still satisfies (b)'s hypothesis with Λ = 60(Λ_loc + 2Λ_σ); the deterministic evaluated head is 60Λ_loc-Lipschitz | `RealizedSatisfiesB`, `RealizedDeterministicLipschitz` | `Statement/Realized.lean` |
| Threshold lemma: packing bound, and response > 2δ/3 with L ≥ 4 ⇒ the count bound never binds; in terms of `cellCount` and `Separated`: M ≤ C_α(L) for every ε and α ≤ 1 | `PackingBound`, `ThresholdLemma`, `CountBoundNeverBinds`, `CountBoundExcludesNothing` | `Statement/Threshold.lean` |
| Without attained Bellman maxima the throughput bound (iii) is empty (ℓ ≤ 0) | `NonAttainmentForcesEllZero`, `NonAttainmentForcesEllZeroGeneral` | `Statement/Throughput*.lean` |
| Note, prop:nofree, small side (charge only lowers the optimum) | `ChargeAntitone`, `ChargeAntitoneInL` | `Statement/NoFree.lean` |
| Note, prop:nofree, large side (zero padding) | `PaddingEmbedding` | `Statement/NoFree.lean` |
| Note, witness bound τ₁ = (1 − 8Nb(L₁)/R)(τ_full − η) | `WitnessScaling`, `WitnessEpisode` | `Statement/NoFree.lean` |
| Note, Corollary cor:air | `CorAir` | `Statement/NoFree.lean` |
| Note, Corollary cor:middle (the combination step) | `CorMiddle` | `Statement/NoFree.lean` |
| Theorem 1(a) at the simulator's constants: κ(i) ≤ R at every tested L, and the two-sided bracket (1 − 8N b(L)/R) τ⁰_t ≤ τ_t ≤ τ⁰_t − 8 b(L) Σ_i λ_t(i)(1 + \|𝒟_t(i)\|) | `ChargeFitsRadio`, `AirtimeBracket` | `Statement/Instance.lean` |
| The seven floor factors 1 − 8N b(L)/R, as exact rationals | `FloorValues` | `Statement/Instance.lean` |
| Theorem 1(b) for the deterministic (evaluated) policy, α = 1: count ≤ ⌈2βΛ√L/ε⌉^L, misses with probability 1 | `PropositionBDeterministic`, `DiracAnsweredWellIff` | `Statement/Deterministic.lean` |
| The trained tanh-Normal head (60·tanh, 60 m disk clip, Gaussian noise) satisfies the W₁-Lipschitz hypothesis of (b) with Λ = 60(Λ_loc + 2Λ_σ) | `TanhNormalLipschitz`, `TanhNormalSatisfiesB` | `Statement/Receivers.lean` |

**Not machine-checked:**
* That a trained MADRE policy satisfies any hypothesis: the value of Λ, a bound β on the
  messages, the 3ε separation of optimal moves at a common ω, the visiting weight w. These are
  modelling assumptions about the simulator and the trained networks.
* The definition of "optimal move" (via Q*) in Theorem 1(b): `Aopt` there is an arbitrary
  family of sets, so (b) holds for any definition. In `TheoremFloorIII` the optimal moves are
  defined from Q*, V*.
* The informal parts of prop:nofree ("the model contains scenarios in which ..."): only its
  two mechanisms are proved. cor:middle is proved as a logical step from its three ingredients;
  that the processes produced by trained pairs satisfy the hypotheses of thm:floor (iii) with one
  common w is a modelling assumption.
* The other direction of complementary slackness ("λ_t(i) > 0 once the radio is busy") is
  false in degenerate cases and is not claimed.
* Theorem thm:floor (iii) is proved twice. `TheoremFloorIII` works on a FINITE probability space
  with a FINITE state space, where the receiver-law hypothesis forces finitely supported receiver
  laws (`ReceiverFiniteSupport`), so it covers discrete-move and deterministic receivers only.
  `TheoremFloorIIIGeneral` works on a GENERAL probability space and covers any receiver law
  (Gaussian, tanh-Normal, ...). It needs these side conditions, listed in its file header:
  measurable s_t, a_t, ω_t, recv and level sets {o(s) = o}; integrability of V*_t(s_t),
  Q*_t(s_t,a_t), r(s_t,a_t) and of the charged throughput; the note's measurability of
  {(ω, x) : dist(x, A*(o,ω)) ≥ ε} and of the miss probability in ω; T > 0. The Markov property
  enters as 𝔼[V*_{t+1}(s_{t+1})] = 𝔼[NV_t(s_t,a_t)] with NV_t = ∫ V*_{t+1} dκ; `MarkovOfKernel`
  proves this from a Markov transition kernel κ for measurable, bounded V*_{t+1}. Both
  versions use a DETERMINISTIC encoder (as g_θ in the paper); the note also allows a randomized
  encoder.
* The W₁-Lipschitz condition is proved for the trained tanh-Normal head (`TanhNormalSatisfiesB`)
  and for any deterministic receiver, GIVEN Lipschitz constants Λ_loc, Λ_σ of the network's
  location and scale outputs in the message. Those constants of the trained network are not
  computed in Lean. The note's softmax-over-directions case is prose only.
* The theory note's simplified scenario (rate-distortion, Section 6) is not formalized.

## What the two bounds say for the trained MADRE policies

**(a), with the floor, is the quantitative statement.** At the simulator's constants
(R = 191 406.25 bit/s, N = 8 drones, one message round per 1 s step), `AirtimeBracket` proves, at
every tested L and at any positions,

  (1 − 8N b(L)/R) τ⁰_t ≤ τ_t ≤ τ⁰_t − 8 b(L) Σ_i λ_t(i)(1 + |𝒟_t(i)|),

and `FloorValues` gives the floor factors exactly:

| L | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| b(L) (bytes on air) | 131 | 198 | 262 | 460 | 786 | 1508 | 2882 |
| 1 − 8N b(L)/R | 0.956 | 0.934 | 0.912 | 0.846 | 0.737 | 0.496 | 0.036 |

So at L = 64 the message exchange costs at most 15.4 % of the uncharged optimum at any positions,
while at L = 512 it can take up to 96.4 % of a loaded relay radio. `ChargeFitsRadio` proves
κ(i) ≤ R at every tested L (largest κ(i) = 184 448 bit/s < R), which is the hypothesis of
`LPOptimumExists`, and it means the scorer's clip `max(R − κ, 0)` never fires in these runs.

**(b) is a structural bound and does not bind for the trained policies.** It is correct and
machine-checked, and it is the right structural reason that a bounded message read by a smooth
receiver can carry only finitely many distinct instructions. But at the tested sizes it gives no
numbers for the trained policies (`notes/planning/BOUND_B_QUANT_oct17.md` in the thesis
workspace):
* M is capped by the move disk. Optimal moves pairwise 3ε apart in the disk of radius 60 m number
  at most P(ε) = 1681, 441, 81, 19, 4, 2, 1 at ε = 1, 2, 5, 10, 25, 40, 50 m.
* The **threshold lemma**, now machine-checked (`Statement/Threshold.lean`): if the receiver's
  response range x = 2βΛ√L exceeds 2δ/3 = 40 m and L ≥ 4, then for every ε, every set of moves in
  the 60 m disk pairwise 3ε apart has at most ⌈x/ε⌉^L = C_1(L) ≤ C_α(L) elements
  (`CountBoundNeverBinds`; for ε ≤ 40 m strictly, via the packing bound (1 + 2δ/(3ε))² and
  `ThresholdLemma`). Note that 2^L alone would not suffice (2^8 = 256 < 1681 at ε = 1 m).
* The measured response range of every trained checkpoint is 59.5 to 85.8 m > 40 m (a
  measurement, in `BOUND_B_QUANT_oct17.md`, not proved). Given it, `CountBoundExcludesNothing`
  proves, in terms of `cellCount` and `Separated`, that M ≤ C_α(L) for every tested L (all ≥ 8),
  every ε and every α ≤ 1, so the count bound of (b) excludes nothing for these policies.
* (b)'s second clause ("every other one lands at least ε from every optimal move with
  probability ≥ (1 − α)/2") is the complement of "answered well" (`near` and `far` are
  complements, `far_eq_compl`); the content of (b) is the count.
The location of the best message size is therefore measured in the paper's experiments, not
derived from (b).

## Correspondence with the LP the simulator solves

The evaluated LP is built in the code repository (`code/`, commit `3520a2bb`; the files cited
below have no local modifications except `om2g/om2g_config.py`, which is cited at that commit).

| Lean | simulator code | status |
|---|---|---|
| capacity rows Σ_k f_k(v,w) ≤ c(v,w), one per edge | `omnet_module/max_flow_reward.py` 762–769 (one row per edge of `edge_list`, coefficient 1) | matches; pairs that are not edges have no variable there, i.e. c(v,w) = 0 in Lean (check-by-hand item 2) |
| radio row of drone i | `max_flow_reward.py` 770–802 (users have no radio row, line 778): coefficient 1 on every flow at an edge incident to i (`coef = 1.0`, line 795), EXCEPT that with `OM2G_PHYSICAL_METRIC=1` the transmitter's coefficient is 1/p (lines 796–797, p from `_EDGE_P`) | **the paper's cells set `OM2G_PHYSICAL_METRIC=1`** (verified on the cluster: 979 of 1120 deployed cells, including every MADRE `br12rt*` cell and the TarMAC/CommNet cells; evaluations inherit it, and `local_eval.sh` line 66 exports it). So the scored LP is the WEIGHTED LP `P1W` with a = `physWeight`: 1/p on edges drone i transmits on to another DRONE, 1 on edges it transmits on to a USER (`_EDGE_P` is filled only in the drone-drone loop of `_build_edges`, lines 488 and 491, and the lookup at 797 defaults to p = 1), 1 on every edge it receives on. Every result of part (a) is proved for it (`Statement/Weighted.lean`), and `tools/check_scorer_lp.py` checks mechanically that the scorer's LP is exactly this LP (see below). The paper's (P1) as printed is the unit-weight special case (`P1`, `UnitWeightSpecialCase`). The same flag also changes c(v,w) (decode gate, lines 400–406), which Lean takes as data |
| right-hand side R − κ(i), κ(i) = 8 b(L)(1 + \|𝒟_t(i)\|) | `max_flow_reward.py` `_build_b_ub`, 419–460: `control_bps = (1 + n_nb)·bytes·8/interval` (453–454), clipped at 0 (457) | matches with interval = 1 s (`CONTROL_SYNC_INTERVAL_S`, line 316, default 1.0, never overridden); the clip never fires (`ChargeFitsRadio`) |
| R = 191 406.25 | `om2g/om2g_track.py` 293–302 (`OM2G_RADIO_CAP_APP_BPS` = 250 000 · 49/64), exported by `om2g/om2g_config.py` line 283 (at `3520a2bb`) with `OM2G_CAP_MULT` = 1 | matches |
| b(L) | `om2g_track.py` `om2g_message_on_air_bytes`, 408–419: payload 8 + 4(12 + L) bytes, one 8-byte UDP header, 20 + 15 bytes per 96-byte IP fragment; gives 131, 198, 262, 460, 786, 1508, 2882 | matches paper Section V |
| N = 8 | paper Table I; launches set `OM2G_DRONES=8` (the code default at `om2g_track.py` 146 is 5) | matches for the paper's runs |
| 𝒟_t(i) = `nbr i` | `n_nb[j]` counts drones i with a positive-capacity (decodable) directed edge (i, j) in the LP, lines 445–452 | **wording**: in the scorer 𝒟_t(i) is the DECODABLE-NEIGHBOUR set; the paper says "drones whose message reached i", a subset. The Lean proofs take `nbr` as arbitrary data, so nothing in them depends on which set is meant |
| two charging conventions | user flows: per DELIVERED bit, coefficient 1 at the receiver and 1/p (drone-drone) or 1 (drone-to-user) at the transmitter, comments at lines 780–794; control plane: κ(i) = 8 b(L)(1 + \|𝒟_t(i)\|) per DECODABLE neighbour, charged whether or not the message is actually delivered | both are what `P1W` + `charge` model: flows enter the radio row with their weights, κ(i) is a constant right-hand-side charge |
| flow conservation with σ_k = +1 at the source, −1 at the sink, 0 elsewhere | `max_flow_reward.py` 816–858 (source balance, sink balance, transit nodes) | matches |
| objective Σ_k τ(k) | lines 749–760, one segment of constant slope (`_build_objective_segments`, 589–605) | matches up to a positive scale factor |
| 0 ≤ τ(k) ≤ D | column bounds 907–912 (`col_lower = 0`, `col_upper = tau_cap`), D = `OM2G_TAU_CAP_BPS` = voice rate × `OM2G_TAU_MULT` (`om2g_config.py` 282 at `3520a2bb`) | matches |
| tanh-Normal head, 60 m step, 60 m disk | BenchMARL MAPPO `use_tanh_normal: True` (`src/conf/algorithm/padre_mappo_norm*.yaml`), action box [−1, 1]² (`omnet_module/omnet_env.py` 1714), × `OMNET_STEP_M` = 60 (`omnet_env.py` 2285–2294, `om2g_track.py` 198), radial clip to `maxSpeed` = 60 mps (`omnet_module/RLControlledMobility.cc` 50–54, `omnet_smoke/om2g.ini` 137) | matches `tanhMove` in `Statement/Receivers.lean` |
| deterministic evaluation | `evaluation_deterministic_actions: True` is BenchMARL's library default (`benchmarl/conf/experiment/base_experiment.yaml` line 99), not set in this repository; torchrl's deterministic sample of `TanhNormal` is the tanh of the location | covered by `PropositionBDeterministic` with Λ = 60Λ_loc (`RealizedDeterministicLipschitz`) |
| torchrl clamp and wall reflection | `torchrl/modules/distributions/utils.py` `_SafeTanhNoEps` (tanh clamped to ±(1 − float resolution)); `RLControlledMobility.cc` line 64 `handleIfOutside(REFLECT, …)` | modelled in `realizedMove` (`Statement/Realized.lean`) |

## Mechanical check of the scorer's LP (Python)

`tools/check_scorer_lp.py` imports the scorer `omnet_module/max_flow_reward.py` READ-ONLY from
the code repository. For each tested L it runs a fresh process with the paper's environment
(`OM2G_PHYSICAL_METRIC=1`, 8 drones, compressed arm, R, b(L), sync interval and demand cap taken
from `om2g/om2g_track.py`) and five random layouts (fixed seeds). It reads back the exact LP that
HiGHS solved (`Highs.getLp()`), rebuilds the LP independently from the `P1EW` definition
(the scorer's positive-capacity edges and capacities; weights 1/p on drone-drone transmit and 1
elsewhere; κ(i) = 8 b(L)(1 + |𝒟(i)|)), and asserts that the objective, column bounds,
inequality rows and equality rows coincide (up to row order, the sign of equality rows and the
scorer's unit scale) and that both optima are equal (reference solved with scipy/HiGHS).
Run it from this directory:

```
python tools/check_scorer_lp.py
```

Result (Oct 17): **PASS for every L ∈ {8, …, 512}** on the full LP. The optima are nontrivial:
demand-bound at small L, and lower at L = 256 and 512, where the charge binds.

## Files

| file | content |
|------|---------|
| `MadreProof/Statement.lean` | (P1), its dual, Theorem 1 (a), (b), strong duality, optimum existence, complementary slackness |
| `MadreProof/Statement/Throughput.lean` | finite-horizon process, Q*, V*, Lemma lem:pdl, Theorem thm:floor (ii), (iii) |
| `MadreProof/Statement/NoFree.lean` | prop:nofree mechanisms, witness bound, cor:air, cor:middle |
| `MadreProof/Proofs/Airtime.lean` | weak LP duality for (P1), with the slack terms |
| `MadreProof/Proofs/Farkas.lean` | conic Carathéodory, closedness of finitely generated cones, Farkas' lemma |
| `MadreProof/Proofs/StrongDuality.lean` | LP strong duality for (P1), existence of its optimum, proofs of (a) and complementary slackness |
| `MadreProof/Proofs/Information.lean` | coupling lemma, cube covering, pigeonhole, proof of (b) |
| `MadreProof/Proofs/CommonOmega.lean` | the common-ω reading of (b), from (b) with a one-point receiver-state type |
| `MadreProof/Proofs/Throughput.lean` | proofs of lem:pdl and thm:floor (ii), (iii) |
| `MadreProof/Proofs/NoFree.lean` | proofs of the prop:nofree mechanisms, witness, cor:air, cor:middle |
| `MadreProof/Statement/ThroughputGeneral.lean`, `Proofs/ThroughputGeneral.lean` | lem:pdl and thm:floor (iii) on a general probability space; the Markov hypothesis from a transition kernel |
| `MadreProof/Statement/EdgeSet.lean`, `Proofs/EdgeSet.lean` | (P1) with an explicit edge set; equivalence with the all-pairs LP; Theorem 1(a) for it |
| `MadreProof/Statement/Weighted.lean`, `Proofs/Weighted{Airtime,StrongDuality,Instance,EdgeSet,Sanity}.lean` | (P1) with a weighted radio row (the scorer's LP) and every part-(a) result for it; the unit LP as special case |
| `MadreProof/Statement/Realized.lean`, `Proofs/Realized.lean` | the realized move (clamp, reflection) and its Lipschitz bounds |
| `MadreProof/Statement/Threshold.lean`, `Proofs/Threshold.lean` | packing bound, threshold lemma, and `CountBoundExcludesNothing` |
| `tools/check_scorer_lp.py` | Python check that the scorer's LP equals `P1EW` (see above) |
| `MadreProof/Proofs/Sanity.lean` | non-vacuity: a tiny LP where every hypothesis of (a) holds with λ > 0 and the bound is tight; a message-dependent Dirac receiver satisfying the W₁-Lipschitz hypothesis of (b); W₁(P,P) = 0 |
| `MadreProof/Statement/Instance.lean`, `Proofs/Instance.lean` | the simulator's constants R, N, b(L); κ(i) ≤ R; the two-sided airtime bracket; the floor factors |
| `MadreProof/Statement/Deterministic.lean`, `Proofs/Deterministic.lean` | Theorem 1(b) for a deterministic receiver with α = 1 |
| `MadreProof/Statement/Receivers.lean`, `Proofs/Receivers.lean` | the tanh-Normal head is W₁-Lipschitz in the message (same-noise coupling; tanh and the disk clip are 1-Lipschitz; 𝔼\|ξ\| bound for N(0, 1)) |
| `MadreProof/Check.lean` | `#print axioms` for every main theorem |

Lean checks the proofs. A reader only has to check that the statement files say what the paper
and the note say. The list below is everything that has to be compared by hand.

## What you must check by hand

### Part (a): `PropositionA`, `LPStrongDuality`, `LPOptimumExists`, `ComplementarySlackness`

1. **(P1) is transcribed correctly.** `P1.Feasible κ f τ` has five conjuncts, the five lines of (P1)
   in order: edge capacities ∑_k f_k(v,w) ≤ c(v,w); flow conservation
   ∑_w f_k(v,w) − ∑_w f_k(w,v) = σ_k(v) τ(k); radio rows ∑_{e ∈ edgesAt i} ∑_k f_k(e) ≤ R − κ(i);
   0 ≤ τ(k) ≤ D; f ≥ 0. The objective is ∑_k τ(k) (`P1.values`). σ_k(v) and the sets
   "e_{v,w} at i" (`edgesAt i`) are arbitrary data.
2. **Edges.** `P1` has a flow variable and a capacity row for every ordered pair (v,w) of nodes;
   the paper has them for e_{v,w} ∈ E only. This is now machine-checked: `P1E`
   (`Statement/EdgeSet.lean`) is (P1) with an explicit edge set, `EdgeSetValuesEq` proves its
   feasible values equal those of the all-pairs LP `P1E.toP1` (c = 0 off E), and
   `PropositionAEdgeSet` states Theorem 1(a) directly for `P1E` with an optimal dual of
   `P1E` (one μ(e) per edge). The reader checks `P1E.Feasible` and `P1E.DualFeasible` instead.
3. **The dual of (P1)** (`P1.DualFeasible`, `P1.dualObj`). Multipliers μ(v,w) ≥ 0 (capacity rows),
   ν(v,k) free (flow rows), λ(i) ≥ 0 (`lam`, radio rows), ρ(k) ≥ 0 (rows τ(k) ≤ D).
   Dual constraint of f_k(v,w): μ(v,w) + ν(v,k) − ν(w,k) + ∑_{i : (v,w) ∈ edgesAt i} λ(i) ≥ 0.
   Dual constraint of τ(k): ρ(k) − ∑_v σ_k(v) ν(v,k) ≥ 1.
   Dual objective: ∑ μ c + ∑_i λ(i)(R − κ(i)) + ∑_k ρ(k) D.
4. **"Optimal dual solution"** (`P1.DualOptimal`): dual feasible, with the least dual objective
   among all dual-feasible points. `LPStrongDuality` proves that one exists and that its value is
   the optimal value τ⁰ of (P1) (nothing about duality is assumed any more).
5. **κ(i) = 8 b(L) (1 + |𝒟_t(i)|)** (`charge b nbr i`), with `nbr i` the set 𝒟_t(i).
6. **Hypotheses of (a).** `τ` is the optimal value of (P1) with the charge, `τ0` that of (P1) at
   κ = 0, `lam` the radio part of an optimal dual solution at κ = 0. `LPOptimumExists` proves
   that the optimal values exist when c ≥ 0, D ≥ 0 and κ(i) ≤ R.
7. **Complementary slackness**: for an optimal (f, τ) of (P1) with charge κ and an optimal dual
   solution, a strictly slack radio row of drone i forces λ(i) = 0.

### Part (b): `PropositionB`, `PropositionBCommonOmega`

8. **Moves** live in any metric space `X` with its Borel σ-algebra (the paper: a disk in ℝ²).
9. **Messages** are `Fin L → ℝ`, the range [-β,β]^L is `box L β`, and their distance is the
   Euclidean one, `msgDist m m' = √(∑_j (m_j − m'_j)²)` (the √L in C_α(L) comes from this).
10. **W₁** (`W1`) is the infimum over all couplings π of P and Q (measures on X × X with marginals
    P and Q) of the Lebesgue integral ∫ dist(a,a') dπ, in [0, ∞]. On ℝ² (any separable space) this
    is the textbook W₁; elsewhere it can only be smaller, which only weakens the hypothesis.
11. **The receiver** `π m ω` is a probability measure, and for every ω and all m, m' in
    [-β,β]^L, W₁(π(·|m,ω), π(·|m',ω)) ≤ Λ ‖m − m'‖. `Sanity.lean` shows a receiver that depends on
    m and satisfies this (the Dirac law at the received value).
12. **Observations** are a finite type `O` (M elements). The **optimal moves** `Aopt o ω` are
    arbitrary sets. If some `Aopt o ω` is empty, o counts as never answered well and the second
    conclusion is vacuous for it ("at least ε from every optimal move" holds trivially), so the
    reader must check that the paper's optimal-move sets are nonempty (the note assumes the
    maxima defining them are attained).
13. **Separation** (`Separated`): every optimal move of o is at least 3ε from every optimal move of
    o' at ω. The hypothesis is that every two observations are separated at some ω (which may
    differ between pairs). The reading "one ω common to all M" is `PropositionBCommonOmega`,
    derived from `PropositionB` by taking the receiver-state type to be a single point ω₀; it
    carries both conclusions at ω₀ and the inclusion "answered well at every such ω ⇒ at ω₀".
14. **"Such ω"** (`SuchOmega`) and **"answered well"** (`AnsweredWell`): at every ω at which o is
    separated from some other observation, the probability of landing strictly within ε of an
    optimal move is strictly above (1 + α)/2.
15. **Constants**: 0 < β, 0 < Λ, 0 < α < 1, 0 < ε. C_α(L) = `⌈2 * β * Λ * √L / (α * ε)⌉₊ ^ L`.
    Λ > 0 and β > 0 are needed (with Λ = 0 or β = 0 the formula gives C_α(L) = 0 while one
    observation can be answered well).

### The note's throughput results: `Statement/Throughput.lean`

16. **Setting (narrower than the note).** Finite probability space `FinProb Ξ`, finite state
    space `S`, arbitrary joint moves `𝔸`, `T` steps, Lean step t = 0..T−1 being the note's
    step t+1. The only dynamic property used (`Process.markov`) is
    𝔼[φ(s_{t+1})] = 𝔼[∑_{s'} Ptr(s_t,a_t,s') φ(s')], implied by the note's Markov property.
17. **Q*, V*** (`OptValues`): the note's Bellman equations, with V* = sup (IsLUB) of Q*.
18. **Receiver law** (`hrecv` in `TheoremFloorIII`): ℙ(recv(a_t) ∈ B(s_t, ω_t)) =
    𝔼[h(B(s_t,ω_t) | g(o_t), ω_t)] for every measurable-set-valued B, which is what the note's
    "a_t(j) ~ h(·|m_t, ω_t) given (s_t, ω_t, m_t)" gives.
19. **Optimal-move sets** `Aopt o ω` ⊇ A*_t(s_t) whenever o_t = o, ω_t = ω (the note's A*(o,ω) is
    such a family); miss probability measurable in ω (the note's assumption); ε-separated family
    `F` with measurable separating sets `Ωsep` and ν(Ωsep o o') ≥ 1; visiting hypothesis
    `hvisit` = the note's Assumption ass:visit, occ({o} × B) ≥ w ν(B), occ = (1/T)∑_t ℙ((o_t,ω_t) ∈ ·).
20. **ℓ**: any 0 ≤ ℓ with ℓ ≤ V*_t(s) − Q*_t(s,a) whenever o(s) ∈ F and the receiver's move is at
    least ε from A*_t(s) (the note's ℓ(ε) is the largest such). **Hand-check: the Bellman maxima
    must be attained** (the note assumes this). If at some step and some state with o(s) ∈ F the
    maximum is not attained, A*_t(s) is empty and the hypothesis forces ℓ ≤ 0
    (`NonAttainmentForcesEllZero`, `…General`), so both versions of (iii) are then empty.
21. **thr**: the charged throughput of step t is at most r(s_t,a_t) (the note's τ_{t+1}(κ) ≤ τ_{t+1}(0)).

### `Statement/NoFree.lean`

22. `WitnessScaling` needs 0 ≤ κ(i) ≤ K ≤ R and R > 0 (with K = 8 N b(L₁)).
23. `CorMiddle`: `A L` stands for the set of episode throughputs of the pairs at size L; its
    hypotheses (uniform bounds at L_min and L_max below τ₁, a pair at L₁ reaching τ₁) are what
    thm:floor (iii), cor:air and the witness bound give.

### `Statement/Instance.lean`, `Statement/Deterministic.lean`, `Statement/Receivers.lean`

24. The constants `simR` = 250000·49/64, `simN` = 8 and `onAirBytes L` are those of the code
    and the paper (correspondence table above). `AirtimeBracket` additionally assumes
    c ≥ 0, D ≥ 0 and 𝒟_t(i) ⊆ 𝒟 ∖ {i} (`i ∉ nbr i`).
25. `PropositionBDeterministic`: the receiver is a function φ(m, ω) that is Λ-Lipschitz in m on
    [−β, β]^L; "answered well" means its move is strictly within ε of an optimal move at every
    such ω. `DiracAnsweredWellIff` shows this is `AnsweredWell` of `PropositionB` for the Dirac
    receiver at any α ∈ (−1, 1).
26. `tanhMove loc σ ξ` = diskClip₆₀(60 · tanh(loc + σ ⊙ ξ)) componentwise, `stdGauss2` = N(0, I₂).
    The hypotheses are Euclidean Lipschitz bounds on m ↦ loc(m, ω) and m ↦ σ(m, ω). If the
    library applies a further 1-Lipschitz map to loc (torchrl's optional `tanh_loc`), it can be
    absorbed into loc. `realizedMove` (`Statement/Realized.lean`) adds torchrl's clamp of tanh to
    [−c, c] and the arena reflection into [lo, hi]² of the new position; **assumption: the
    receiver state ω carries the drone's current position** (`pos ω`), as the observation does.
    The realized move satisfies the same Λ = 60(Λ_loc + 2Λ_σ).

27. `TheoremFloorIIIGeneral`: `MProcess` (probability measure P, s_t, a_t, r) and `MOptValues`
    (Bellman equations as in the finite version, `markov`, integrability). `hrecv` says the joint
    law of (s_t, ω_t, recv(a_t)) is law(s_t, ω_t) followed by h(·|g(o_t), ω_t): for every
    measurable B ⊆ S × Ω × X, P((s_t, ω_t, recv(a_t)) ∈ B) = 𝔼[h(B_{(s_t,ω_t)} | g(o_t), ω_t)].
    The visiting hypothesis is Assumption ass:visit with P.
28. `MarkovOfKernel`: κ is a Markov kernel from S × 𝔸 to S, the joint law of
    ((s_t, a_t), s_{t+1}) is law(s_t, a_t) ⊗ κ, and V is measurable and bounded.
29. `P1W` (`Statement/Weighted.lean`) is `P1` with the radio row Σ_{v,w} a_i(v,w) Σ_k f_k(v,w)
    ≤ R − κ(i) and the dual constraint of f_k(v,w) carrying Σ_i a_i(v,w) λ(i); no condition on a.
    `physWeight drone p` is the scorer's a: 1/p(v,w) when v is drone i (transmit), 1 when w is
    drone i (receive), 0 otherwise.
