/-
# (P1) with a WEIGHTED radio row: formal statement

Statements only (proofs in `MadreProof/Proofs/Weighted*.lean`). The scorer of the paper's runs
(`omnet_module/max_flow_reward.py` lines 770–802, with `OM2G_PHYSICAL_METRIC=1`, which the paper's
cells export) charges the radio row of drone i with coefficient 1/p(i,w) on the flow of every
drone-drone edge (i, w) that i TRANSMITS on (p = per-frame delivery probability, `_EDGE_P`),
and 1 on edges to users and on every edge (v, i) it receives on (see `physWeight`). `P1W` therefore replaces the unit coefficients of `P1` by arbitrary
weights a_i(v,w) in the radio row:

  ∑_{v,w} a_i(v,w) ∑_k f_k(v,w) ≤ R − κ(i).

No sign or other condition on the weights is needed by any proof here. The paper's (P1) as
printed is the unit-weight case a_i(v,w) = 1 for (v,w) ∈ edgesAt i and 0 otherwise
(`P1.toW`, `UnitWeightSpecialCase`); the scorer's case is `physWeight`. Every headline theorem
about `P1` is restated for `P1W`; the `P1` theorems are kept and also re-derived from the
weighted ones (`*_fromWeighted` in `Proofs/Weighted.lean`).
-/
import MadreProof.Statement.Instance
import MadreProof.Statement.EdgeSet

namespace Madre

/-- (P1) with a weighted radio row. -/
structure P1W (V C D : Type) where
  c : V → V → ℝ
  σ : C → V → ℝ
  R : ℝ
  demand : ℝ
  /-- a_i(v,w): the radio-row coefficient of the flow on (v,w) for drone i. -/
  a : D → V → V → ℝ

variable {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]

/-- Feasibility: as `P1.Feasible`, with the weighted radio row. -/
def P1W.Feasible (P : P1W V C D) (κ : D → ℝ) (f : C → V → V → ℝ) (τ : C → ℝ) : Prop :=
  (∀ v w, ∑ k, f k v w ≤ P.c v w) ∧
  (∀ v k, ∑ w, f k v w - ∑ w, f k w v = P.σ k v * τ k) ∧
  (∀ i, ∑ v, ∑ w, P.a i v w * ∑ k, f k v w ≤ P.R - κ i) ∧
  (∀ k, 0 ≤ τ k ∧ τ k ≤ P.demand) ∧
  (∀ k v w, 0 ≤ f k v w)

def P1W.values (P : P1W V C D) (κ : D → ℝ) : Set ℝ :=
  {x | ∃ f τ, P.Feasible κ f τ ∧ x = ∑ k, τ k}

/-- The LP dual: the dual constraint of f_k(v,w) carries ∑_i a_i(v,w) λ(i). -/
def P1W.DualFeasible (P : P1W V C D) (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ)
    (ρ : C → ℝ) : Prop :=
  (∀ v w, 0 ≤ μ v w) ∧ (∀ i, 0 ≤ lam i) ∧ (∀ k, 0 ≤ ρ k) ∧
  (∀ k v w, 0 ≤ μ v w + ν v k - ν w k + ∑ i, P.a i v w * lam i) ∧
  (∀ k, 1 ≤ ρ k - ∑ v, P.σ k v * ν v k)

def P1W.dualObj (P : P1W V C D) (κ : D → ℝ) (μ : V → V → ℝ) (lam : D → ℝ) (ρ : C → ℝ) : ℝ :=
  ∑ v, ∑ w, μ v w * P.c v w + ∑ i, lam i * (P.R - κ i) + ∑ k, ρ k * P.demand

def P1W.DualOptimal (P : P1W V C D) (κ : D → ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) : Prop :=
  P.DualFeasible μ ν lam ρ ∧
    ∀ μ' ν' lam' ρ', P.DualFeasible μ' ν' lam' ρ' → P.dualObj κ μ lam ρ ≤ P.dualObj κ μ' lam' ρ'

/-- The paper's (P1) as a weighted LP: weight 1 on the edges at i, 0 elsewhere. -/
def P1.toW (P : P1 V C D) : P1W V C D where
  c := P.c
  σ := P.σ
  R := P.R
  demand := P.demand
  a := fun i v w => if (v, w) ∈ P.edgesAt i then 1 else 0

/-- The scorer's weights with `OM2G_PHYSICAL_METRIC=1` (`max_flow_reward.py` lines 795–797):
the radio row of drone i (node `drone i`) has coefficient 1/p(v, w) on an edge (v, w) that i
TRANSMITS on to another DRONE (`isDrone w`), because the delivery probabilities `_EDGE_P` are
recorded only in the drone-drone loop of `_build_edges` (lines 488, 491; lookups default to
p = 1); coefficient 1 on an edge i transmits on to a USER and on every edge i receives on;
0 elsewhere (edges absent from the LP have c = 0 and carry no flow). Caveat (scorer, not Lean):
`_EDGE_P` is never cleared, so in the leave-one-out LPs of the difference reward a stale entry
can put 1/p on a drone-to-user edge; that LP is still a `P1W`, with other weights. -/
noncomputable def physWeight (drone : D → V) (isDrone : V → Bool) (p : V → V → ℝ) (i : D)
    (v w : V) : ℝ :=
  if v = drone i then (if isDrone w then 1 / p v w else 1) else if w = drone i then 1 else 0

/-- **The unit-weight LP is the special case**: same feasible points, same dual feasible
points, same dual objective, hence the same values and the same optimal duals. -/
def UnitWeightSpecialCase : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V] (P : P1 V C D),
    (∀ κ f τ, P.Feasible κ f τ ↔ P.toW.Feasible κ f τ) ∧
    (∀ μ ν lam ρ, P.DualFeasible μ ν lam ρ ↔ P.toW.DualFeasible μ ν lam ρ) ∧
    (∀ κ μ lam ρ, P.dualObj κ μ lam ρ = P.toW.dualObj κ μ lam ρ)

/-- **LP strong duality, weighted.** -/
def LPStrongDualityW : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1W V C D) (κ : D → ℝ) (τ0 : ℝ),
    IsGreatest (P.values κ) τ0 →
    ∃ (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
      P.DualOptimal κ μ ν lam ρ ∧ P.dualObj κ μ lam ρ = τ0

/-- **The optimum exists, weighted** (c ≥ 0, D ≥ 0, κ(i) ≤ R). -/
def LPOptimumExistsW : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1W V C D) (κ : D → ℝ),
    (∀ v w, 0 ≤ P.c v w) → 0 ≤ P.demand → (∀ i, κ i ≤ P.R) →
    ∃ τ0, IsGreatest (P.values κ) τ0

/-- **Proposition 1(a), weighted radio row.** -/
def PropositionAW : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1W V C D) (b : ℝ) (nbr : D → Finset D) (τ τ0 : ℝ)
    (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
    IsGreatest (P.values (charge b nbr)) τ →
    IsGreatest (P.values 0) τ0 →
    P.DualOptimal 0 μ ν lam ρ →
    τ ≤ τ0 - 8 * b * ∑ i, lam i * (1 + (nbr i).card)

/-- **Complementary slackness, weighted.** -/
def ComplementarySlacknessW : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1W V C D) (κ : D → ℝ) (f : C → V → V → ℝ) (τ : C → ℝ)
    (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
    P.Feasible κ f τ →
    IsGreatest (P.values κ) (∑ k, τ k) →
    P.DualOptimal κ μ ν lam ρ →
    ∀ i, ∑ v, ∑ w, P.a i v w * ∑ k, f k v w < P.R - κ i → lam i = 0

/-- **Witness scaling, weighted** (uniform scaling of all flows). -/
def WitnessScalingW : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1W V C D) (κ : D → ℝ) (K x : ℝ),
    0 < P.R → (∀ i, 0 ≤ κ i ∧ κ i ≤ K) → 0 ≤ K → K ≤ P.R →
    x ∈ P.values 0 → (1 - K / P.R) * x ∈ P.values κ

/-- **Charge antitone, weighted.** -/
def ChargeAntitoneW : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1W V C D) (κ κ' : D → ℝ) (τ τ' : ℝ),
    (∀ i, κ i ≤ κ' i) → IsGreatest (P.values κ) τ → IsGreatest (P.values κ') τ' → τ' ≤ τ

/-- **Two-sided airtime bracket at the simulator's constants, weighted radio row** (in
particular for the scorer's weights `physWeight`); the floor factors are those of
`FloorValues`, unchanged. -/
def AirtimeBracketW : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V] [DecidableEq D]
    (P : P1W V C D) (L : ℕ) (nbr : D → Finset D),
    L ∈ testedL → P.R = simR → Fintype.card D ≤ simN → (∀ i, i ∉ nbr i) →
    (∀ v w, 0 ≤ P.c v w) → 0 ≤ P.demand →
    (∃ τ, IsGreatest (P.values (charge (onAirBytes L) nbr)) τ) ∧
    (∃ τ0, IsGreatest (P.values 0) τ0) ∧
    ∀ (τ τ0 : ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
      IsGreatest (P.values (charge (onAirBytes L) nbr)) τ →
      IsGreatest (P.values 0) τ0 →
      P.DualOptimal 0 μ ν lam ρ →
      floorFactor L * τ0 ≤ τ ∧
      τ ≤ τ0 - 8 * onAirBytes L * ∑ i, lam i * (1 + (nbr i).card)

/-! ### Explicit edge set with a weighted radio row -/

/-- (P1) with an explicit edge set and weights a_i(e) on the edges. -/
structure P1EW (V C D : Type) where
  E : Finset (V × V)
  c : V × V → ℝ
  σ : C → V → ℝ
  R : ℝ
  demand : ℝ
  a : D → V × V → ℝ

def P1EW.Feasible (P : P1EW V C D) (κ : D → ℝ) (f : C → V × V → ℝ) (τ : C → ℝ) : Prop :=
  (∀ e ∈ P.E, ∑ k, f k e ≤ P.c e) ∧
  (∀ v k, ∑ e ∈ P.E.filter (fun e => e.1 = v), f k e
      - ∑ e ∈ P.E.filter (fun e => e.2 = v), f k e = P.σ k v * τ k) ∧
  (∀ i, ∑ e ∈ P.E, P.a i e * ∑ k, f k e ≤ P.R - κ i) ∧
  (∀ k, 0 ≤ τ k ∧ τ k ≤ P.demand) ∧
  (∀ k, ∀ e ∈ P.E, 0 ≤ f k e)

def P1EW.values (P : P1EW V C D) (κ : D → ℝ) : Set ℝ :=
  {x | ∃ f τ, P.Feasible κ f τ ∧ x = ∑ k, τ k}

def P1EW.DualFeasible (P : P1EW V C D) (μ : V × V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ)
    (ρ : C → ℝ) : Prop :=
  (∀ e ∈ P.E, 0 ≤ μ e) ∧ (∀ i, 0 ≤ lam i) ∧ (∀ k, 0 ≤ ρ k) ∧
  (∀ k, ∀ e ∈ P.E, 0 ≤ μ e + ν e.1 k - ν e.2 k + ∑ i, P.a i e * lam i) ∧
  (∀ k, 1 ≤ ρ k - ∑ v, P.σ k v * ν v k)

def P1EW.dualObj (P : P1EW V C D) (κ : D → ℝ) (μ : V × V → ℝ) (lam : D → ℝ) (ρ : C → ℝ) : ℝ :=
  ∑ e ∈ P.E, μ e * P.c e + ∑ i, lam i * (P.R - κ i) + ∑ k, ρ k * P.demand

def P1EW.DualOptimal (P : P1EW V C D) (κ : D → ℝ) (μ : V × V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) : Prop :=
  P.DualFeasible μ ν lam ρ ∧
    ∀ μ' ν' lam' ρ', P.DualFeasible μ' ν' lam' ρ' → P.dualObj κ μ lam ρ ≤ P.dualObj κ μ' lam' ρ'

/-- The all-pairs weighted LP with c = 0 and a = 0 off E. -/
def P1EW.toW (P : P1EW V C D) : P1W V C D where
  c := fun v w => if (v, w) ∈ P.E then P.c (v, w) else 0
  σ := P.σ
  R := P.R
  demand := P.demand
  a := fun i v w => if (v, w) ∈ P.E then P.a i (v, w) else 0

/-- Same feasible values as the all-pairs weighted LP. -/
def EdgeSetValuesEqW : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1EW V C D) (κ : D → ℝ), P.values κ = P.toW.values κ

/-- **Proposition 1(a) for the explicit-edge, weighted LP.** -/
def PropositionAEdgeSetW : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1EW V C D) (b : ℝ) (nbr : D → Finset D) (τ τ0 : ℝ)
    (μ : V × V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
    IsGreatest (P.values (charge b nbr)) τ →
    IsGreatest (P.values 0) τ0 →
    P.DualOptimal 0 μ ν lam ρ →
    τ ≤ τ0 - 8 * b * ∑ i, lam i * (1 + (nbr i).card)

end Madre
