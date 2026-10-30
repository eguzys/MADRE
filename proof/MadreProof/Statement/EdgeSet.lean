/-
# (P1) with an explicit edge set: formal statement

Statements only (proofs in `MadreProof/Proofs/EdgeSet.lean`). `P1E` is (P1) exactly as the
paper writes it: flow variables f_k(e) and capacity rows only for the edges e = (v, w) in a
finite set E ⊆ V × V. The rest of the project uses `P1`, which has a flow variable for every
ordered pair of nodes; this file proves that the two agree (`P1E.toP1`, the all-pairs LP with
c = 0 off E) and states Proposition 1(a) directly for `P1E`, so the README's hand-check
"non-edges via c(v,w) = 0" is machine-checked.
-/
import MadreProof.Statement

namespace Madre

/-- (P1) with an explicit edge set `E`. -/
structure P1E (V C D : Type) where
  /-- The edge set E: ordered pairs (v, w) with an edge e_{v,w}. -/
  E : Finset (V × V)
  /-- c(e), the capacity of edge e (used only for e ∈ E). -/
  c : V × V → ℝ
  σ : C → V → ℝ
  R : ℝ
  demand : ℝ
  /-- The edges "at i" of the radio row of drone i. -/
  edgesAt : D → Finset (V × V)

variable {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]

/-- Feasibility of (f, τ), f_k(e) defined for e ∈ E: capacity rows, flow conservation (out-edges
minus in-edges of v), radio rows, demand bounds, nonnegativity. -/
def P1E.Feasible (P : P1E V C D) (κ : D → ℝ) (f : C → V × V → ℝ) (τ : C → ℝ) : Prop :=
  (∀ e ∈ P.E, ∑ k, f k e ≤ P.c e) ∧
  (∀ v k, ∑ e ∈ P.E.filter (fun e => e.1 = v), f k e
      - ∑ e ∈ P.E.filter (fun e => e.2 = v), f k e = P.σ k v * τ k) ∧
  (∀ i, ∑ e ∈ P.E ∩ P.edgesAt i, ∑ k, f k e ≤ P.R - κ i) ∧
  (∀ k, 0 ≤ τ k ∧ τ k ≤ P.demand) ∧
  (∀ k, ∀ e ∈ P.E, 0 ≤ f k e)

/-- The objective values of feasible points. -/
def P1E.values (P : P1E V C D) (κ : D → ℝ) : Set ℝ :=
  {x | ∃ f τ, P.Feasible κ f τ ∧ x = ∑ k, τ k}

/-- Dual feasibility: one multiplier μ(e) ≥ 0 per edge of E, ν free, λ ≥ 0, ρ ≥ 0, one dual
constraint per primal variable f_k(e), e ∈ E, and per τ(k). -/
def P1E.DualFeasible (P : P1E V C D) (μ : V × V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ)
    (ρ : C → ℝ) : Prop :=
  (∀ e ∈ P.E, 0 ≤ μ e) ∧ (∀ i, 0 ≤ lam i) ∧ (∀ k, 0 ≤ ρ k) ∧
  (∀ k, ∀ e ∈ P.E, 0 ≤ μ e + ν e.1 k - ν e.2 k + ∑ i, if e ∈ P.edgesAt i then lam i else 0) ∧
  (∀ k, 1 ≤ ρ k - ∑ v, P.σ k v * ν v k)

/-- The dual objective. -/
def P1E.dualObj (P : P1E V C D) (κ : D → ℝ) (μ : V × V → ℝ) (lam : D → ℝ) (ρ : C → ℝ) : ℝ :=
  ∑ e ∈ P.E, μ e * P.c e + ∑ i, lam i * (P.R - κ i) + ∑ k, ρ k * P.demand

/-- An optimal dual solution of `P1E`. -/
def P1E.DualOptimal (P : P1E V C D) (κ : D → ℝ) (μ : V × V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) : Prop :=
  P.DualFeasible μ ν lam ρ ∧
    ∀ μ' ν' lam' ρ', P.DualFeasible μ' ν' lam' ρ' → P.dualObj κ μ lam ρ ≤ P.dualObj κ μ' lam' ρ'

/-- The all-pairs LP with c = 0 off E (and radio rows restricted to E). -/
def P1E.toP1 (P : P1E V C D) : P1 V C D where
  c := fun v w => if (v, w) ∈ P.E then P.c (v, w) else 0
  σ := P.σ
  R := P.R
  demand := P.demand
  edgesAt := fun i => P.E ∩ P.edgesAt i

/-- **The explicit-edge LP and the all-pairs LP with c = 0 off E have the same feasible
values** (for every charge κ). -/
def EdgeSetValuesEq : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1E V C D) (κ : D → ℝ), P.values κ = P.toP1.values κ

/-- **Proposition 1(a) for the explicit-edge LP.** `lam` is the radio part of an optimal
dual solution of `P1E` at κ = 0. -/
def PropositionAEdgeSet : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1E V C D) (b : ℝ) (nbr : D → Finset D) (τ τ0 : ℝ)
    (μ : V × V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
    IsGreatest (P.values (charge b nbr)) τ →
    IsGreatest (P.values 0) τ0 →
    P.DualOptimal 0 μ ν lam ρ →
    τ ≤ τ0 - 8 * b * ∑ i, lam i * (1 + (nbr i).card)

end Madre
