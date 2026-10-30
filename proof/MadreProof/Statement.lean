/-
# Proposition 1 of the MADRE paper: formal statement

This is the file a reader compares with Proposition 1 of the paper. It contains
definitions and statements only, no proofs: `PropositionA`, `PropositionB` (the paper's
Proposition 1), `LPStrongDuality` and `ComplementarySlackness` (part (a) and the sentence
"λ_t(i) vanishes while drone i has airtime to spare"), and `PropositionBCommonOmega`.
The theory note's Lemma lem:pdl and Theorem thm:floor (ii), (iii) are stated in
`MadreProof/Statement/Throughput.lean`, and prop:nofree, cor:air, cor:middle and the
witness bound in `MadreProof/Statement/NoFree.lean`. The scorer of the paper's runs weights the transmitter's radio row by 1/p
(`OM2G_PHYSICAL_METRIC=1`); that weighted LP, of which `P1` below is the unit-weight special
case, is `P1W` in `MadreProof/Statement/Weighted.lean`, with every part-(a) result restated. The proofs are in
`MadreProof/Proofs/`, and `MadreProof/Check.lean` prints the axioms they use.

Paper symbols, and their names here:

| paper            | here                                                     |
|------------------|----------------------------------------------------------|
| (P1)             | `P1`, `P1.Feasible`, `P1.values`                         |
| c(v,w), σ_k(v)   | `P1.c`, `P1.σ`                                           |
| R, D (demand)    | `P1.R`, `P1.demand`                                      |
| "e_{v,w} at i"   | `P1.edgesAt i` (ordered pairs (v,w))                     |
| f_k(v,w), τ_t(k) | `f k v w`, `τ k`                                         |
| κ(i)             | `charge b nbr i = 8 * b * (1 + |nbr i|)`                 |
| b(L)             | `b` (on-air bytes of one message)                        |
| 𝒟_t(i)           | `nbr i`                                                  |
| τ_t, τ⁰_t        | `τ`, `τ0` (optimal values of (P1) with and without κ)    |
| λ_t(i)           | `lam i`, radio-row part of an optimal dual (`DualOptimal`) |
| [-β,β]^L, ‖·‖    | `box L β`, `msgDist` (Euclidean)                         |
| W₁               | `W1`                                                     |
| π_θ(·|m,ω)       | `π m ω`                                                  |
| g_θ              | `g`                                                      |
| Λ, α, ε, β       | `Λ`, `α`, `ε`, `β`                                       |
| M observations   | the finite type `O`, M = `Fintype.card O`                |
| optimal moves    | `Aopt o ω`                                               |
| C_α(L)           | `cellCount β Λ α ε L`                                    |
| ω                | `ω : Ω`                                                  |
-/
import Mathlib

open MeasureTheory
open scoped ENNReal

namespace Madre

/-! ## Part (a): the airtime bound -/

/-- The data of the linear program (P1) at one fixed time step `t`, with nodes `V`
(drones and users), commodities `C` and drones `D` (one radio row per drone). -/
structure P1 (V C D : Type) where
  /-- `c v w` is the capacity c(v,w) of edge e_{v,w}. -/
  c : V → V → ℝ
  /-- `σ k v` is σ_k(v): +1 at the source of commodity k, -1 at its sink, 0 elsewhere.
  (The proof works for any real values.) -/
  σ : C → V → ℝ
  /-- R, the radio capacity of a drone. -/
  R : ℝ
  /-- D, the demand of one commodity (renamed because 𝒟 is the drone set). -/
  demand : ℝ
  /-- The edges "at i" of the radio row of drone i, as ordered pairs (v,w).
  (The proof works for any choice of these sets.) -/
  edgesAt : D → Finset (V × V)

variable {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]

/-- (f, τ) is feasible for (P1) when the radio row of drone i has right-hand side `R - κ i`.
The five conjuncts are the five constraint lines of (P1), in order. -/
def P1.Feasible (P : P1 V C D) (κ : D → ℝ) (f : C → V → V → ℝ) (τ : C → ℝ) : Prop :=
  (∀ v w, ∑ k, f k v w ≤ P.c v w) ∧
  (∀ v k, ∑ w, f k v w - ∑ w, f k w v = P.σ k v * τ k) ∧
  (∀ i, ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2 ≤ P.R - κ i) ∧
  (∀ k, 0 ≤ τ k ∧ τ k ≤ P.demand) ∧
  (∀ k v w, 0 ≤ f k v w)

/-- The set of objective values ∑_k τ(k) of feasible points of (P1) with charge `κ`. -/
def P1.values (P : P1 V C D) (κ : D → ℝ) : Set ℝ :=
  {x | ∃ f τ, P.Feasible κ f τ ∧ x = ∑ k, τ k}

/-- Feasibility for the LP dual of (P1). Multipliers: `μ v w ≥ 0` (capacity rows),
`ν v k` free (flow-conservation rows), `lam i ≥ 0` (radio rows, λ(i) in the paper),
`ρ k ≥ 0` (rows τ(k) ≤ D). The last two conjuncts are the dual constraints of the
primal variables f_k(v,w) ≥ 0 and τ(k) ≥ 0. -/
def P1.DualFeasible (P : P1 V C D) (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ)
    (ρ : C → ℝ) : Prop :=
  (∀ v w, 0 ≤ μ v w) ∧ (∀ i, 0 ≤ lam i) ∧ (∀ k, 0 ≤ ρ k) ∧
  (∀ k v w, 0 ≤ μ v w + ν v k - ν w k + ∑ i, if (v, w) ∈ P.edgesAt i then lam i else 0) ∧
  (∀ k, 1 ≤ ρ k - ∑ v, P.σ k v * ν v k)

/-- The dual objective of (P1) with charge `κ` (it does not involve ν). -/
def P1.dualObj (P : P1 V C D) (κ : D → ℝ) (μ : V → V → ℝ) (lam : D → ℝ) (ρ : C → ℝ) : ℝ :=
  ∑ v, ∑ w, μ v w * P.c v w + ∑ i, lam i * (P.R - κ i) + ∑ k, ρ k * P.demand

/-- κ(i) = 8 b(L) (1 + |𝒟_t(i)|), with `b` = b(L) and `nbr i` = 𝒟_t(i). -/
def charge (b : ℝ) (nbr : D → Finset D) (i : D) : ℝ := 8 * b * (1 + (nbr i).card)

/-- An optimal dual solution of (P1) with charge `κ`: a dual-feasible point whose dual
objective is the least among all dual-feasible points. -/
def P1.DualOptimal (P : P1 V C D) (κ : D → ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) : Prop :=
  P.DualFeasible μ ν lam ρ ∧
    ∀ μ' ν' lam' ρ', P.DualFeasible μ' ν' lam' ρ' → P.dualObj κ μ lam ρ ≤ P.dualObj κ μ' lam' ρ'

/-- **LP strong duality for (P1)** (proved, not assumed). If (P1) with charge `κ` has
optimal value `τ0`, an optimal dual solution exists and its dual objective equals `τ0`. -/
def LPStrongDuality : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1 V C D) (κ : D → ℝ) (τ0 : ℝ),
    IsGreatest (P.values κ) τ0 →
    ∃ (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
      P.DualOptimal κ μ ν lam ρ ∧ P.dualObj κ μ lam ρ = τ0

/-- **(P1) attains its optimum** (proved): with nonnegative capacities and demand and a charge
κ(i) ≤ R, the feasible set is nonempty (f = 0, τ = 0) and compact, so the optimal
value τ_t of (P1) exists. -/
def LPOptimumExists : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1 V C D) (κ : D → ℝ),
    (∀ v w, 0 ≤ P.c v w) → 0 ≤ P.demand → (∀ i, κ i ≤ P.R) →
    ∃ τ0, IsGreatest (P.values κ) τ0

/-- **Proposition 1(a).** τ_t ≤ τ⁰_t − 8 b(L) ∑_i λ_t(i) (1 + |𝒟_t(i)|).

Hypotheses: `τ` is the optimal value of (P1) with the charge, `τ0` the optimal value of
(P1) at κ = 0, and `lam` is the radio-row part of an optimal dual solution
`(μ, ν, lam, ρ)` of (P1) at κ = 0. (Such a solution exists by `LPStrongDuality`.) -/
def PropositionA : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1 V C D) (b : ℝ) (nbr : D → Finset D) (τ τ0 : ℝ)
    (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
    IsGreatest (P.values (charge b nbr)) τ →
    IsGreatest (P.values 0) τ0 →
    P.DualOptimal 0 μ ν lam ρ →
    τ ≤ τ0 - 8 * b * ∑ i, lam i * (1 + (nbr i).card)

/-- **Complementary slackness, one direction** (the sentence "λ_t(i) vanishes while
drone i has airtime to spare"). `(f, τ)` is an optimal point of (P1) with charge `κ`
(take κ = 0 for the paper's λ_t(i)) and `(μ, ν, lam, ρ)` an optimal dual solution.
If the radio row of drone `i` is slack at `f`, then `lam i = 0`. -/
def ComplementarySlackness : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1 V C D) (κ : D → ℝ) (f : C → V → V → ℝ) (τ : C → ℝ)
    (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
    P.Feasible κ f τ →
    IsGreatest (P.values κ) (∑ k, τ k) →
    P.DualOptimal κ μ ν lam ρ →
    ∀ i, ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2 < P.R - κ i → lam i = 0

/-! ## Part (b): the information bound -/

/-- The message range [-β, β]^L. -/
def box (L : ℕ) (β : ℝ) : Set (Fin L → ℝ) := {m | ∀ j, |m j| ≤ β}

/-- The Euclidean distance ‖m − m'‖ between two messages. -/
noncomputable def msgDist {L : ℕ} (m m' : Fin L → ℝ) : ℝ :=
  Real.sqrt (∑ j, (m j - m' j) ^ 2)

/-- Couplings of two measures `P`, `Q` on the move space: measures on pairs of moves
whose two marginals are `P` and `Q`. -/
def couplings {X : Type} [MeasurableSpace X] (P Q : Measure X) : Set (Measure (X × X)) :=
  {π | π.map Prod.fst = P ∧ π.map Prod.snd = Q}

/-- The Wasserstein distance W₁(P, Q) = inf over couplings π of ∫ dist(a, a') dπ,
valued in [0, ∞]. -/
noncomputable def W1 {X : Type} [MetricSpace X] [MeasurableSpace X] (P Q : Measure X) : ℝ≥0∞ :=
  ⨅ π ∈ couplings P Q, ∫⁻ z, edist z.1 z.2 ∂π

/-- The moves within ε of some move in `A` ("within ε of an optimal move"). -/
def near {X : Type} [MetricSpace X] (ε : ℝ) (A : Set X) : Set X :=
  {a | ∃ a' ∈ A, dist a a' < ε}

/-- The moves at least ε from every move in `A`. -/
def far {X : Type} [MetricSpace X] (ε : ℝ) (A : Set X) : Set X :=
  {a | ∀ a' ∈ A, ε ≤ dist a a'}

/-- C_α(L) = ⌈2 β Λ √L / (α ε)⌉^L. -/
noncomputable def cellCount (β Λ α ε : ℝ) (L : ℕ) : ℕ :=
  ⌈2 * β * Λ * Real.sqrt L / (α * ε)⌉₊ ^ L

/-- At receiver state `ω`, the optimal moves of observations `o` and `o'` are pairwise
at least 3ε apart. -/
def Separated {X O Ω : Type} [MetricSpace X] (Aopt : O → Ω → Set X) (ε : ℝ)
    (o o' : O) (ω : Ω) : Prop :=
  ∀ a ∈ Aopt o ω, ∀ a' ∈ Aopt o' ω, 3 * ε ≤ dist a a'

/-- `ω` is a "such ω" for `o`: the optimal moves of `o` and of some other observation
are 3ε apart at `ω`. -/
def SuchOmega {X O Ω : Type} [MetricSpace X] (Aopt : O → Ω → Set X) (ε : ℝ)
    (o : O) (ω : Ω) : Prop :=
  ∃ o', o' ≠ o ∧ Separated Aopt ε o o' ω

/-- Observation `o` is answered well: at every such ω, the move drawn from
π_θ(· | g_θ(o), ω) is within ε of an optimal move with probability above (1 + α)/2. -/
def AnsweredWell {X O Ω : Type} [MetricSpace X] [MeasurableSpace X] {L : ℕ}
    (π : (Fin L → ℝ) → Ω → Measure X) (g : O → Fin L → ℝ) (Aopt : O → Ω → Set X)
    (α ε : ℝ) (o : O) : Prop :=
  ∀ ω, SuchOmega Aopt ε o ω → (1 + α) / 2 < (π (g o) ω).real (near ε (Aopt o ω))

/-- **Proposition 1(b).** Moves live in any metric space `X` with its Borel σ-algebra
(the paper: a disk in ℝ²). `π m ω` is the law of the move given message `m` and the
rest `ω` of the local state, Λ-Lipschitz in `m` on [-β,β]^L for W₁. `O` is the finite
set of the M sender observations, `Aopt o ω` the set of optimal moves (any sets), and
every two observations are separated at some common ω. Then, for every encoder `g`
into [-β,β]^L, at most C_α(L) observations are answered well, and every other one
lands, at some such ω, at least ε from every optimal move with probability at least
(1 − α)/2. -/
def PropositionB : Prop :=
  ∀ {X Ω O : Type} [MetricSpace X] [MeasurableSpace X] [BorelSpace X] [Finite O]
    (L : ℕ) (β Λ α ε : ℝ), 0 < β → 0 < Λ → 0 < α → α < 1 → 0 < ε →
    ∀ (π : (Fin L → ℝ) → Ω → Measure X), (∀ m ω, IsProbabilityMeasure (π m ω)) →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      W1 (π m ω) (π m' ω) ≤ ENNReal.ofReal (Λ * msgDist m m')) →
    ∀ (Aopt : O → Ω → Set X),
    (∀ o o', o ≠ o' → ∃ ω, Separated Aopt ε o o' ω) →
    ∀ (g : O → Fin L → ℝ), (∀ o, g o ∈ box L β) →
      {o | AnsweredWell π g Aopt α ε o}.ncard ≤ cellCount β Λ α ε L ∧
      ∀ o, ¬ AnsweredWell π g Aopt α ε o →
        ∃ ω, SuchOmega Aopt ε o ω ∧ (1 - α) / 2 ≤ (π (g o) ω).real (far ε (Aopt o ω))

/-- **Proposition 1(b), "one ω common to all M observations" reading.** If every two
observations are separated at one common receiver state `ω₀`, then (i) at most C_α(L) of
them are answered at `ω₀` within ε of an optimal move with probability above (1 + α)/2,
(ii) every other one lands at `ω₀` at least ε from every optimal move with probability at
least (1 − α)/2, and (iii) an observation answered well in the sense of `PropositionB`
(at every such ω) is answered well at `ω₀` as soon as some other observation exists, so the
count in (i) also bounds the paper's "answered at every such ω". -/
def PropositionBCommonOmega : Prop :=
  ∀ {X Ω O : Type} [MetricSpace X] [MeasurableSpace X] [BorelSpace X] [Finite O]
    (L : ℕ) (β Λ α ε : ℝ), 0 < β → 0 < Λ → 0 < α → α < 1 → 0 < ε →
    ∀ (π : (Fin L → ℝ) → Ω → Measure X), (∀ m ω, IsProbabilityMeasure (π m ω)) →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      W1 (π m ω) (π m' ω) ≤ ENNReal.ofReal (Λ * msgDist m m')) →
    ∀ (Aopt : O → Ω → Set X) (ω₀ : Ω),
    (∀ o o', o ≠ o' → Separated Aopt ε o o' ω₀) →
    ∀ (g : O → Fin L → ℝ), (∀ o, g o ∈ box L β) →
      {o | (1 + α) / 2 < (π (g o) ω₀).real (near ε (Aopt o ω₀))}.ncard
        ≤ cellCount β Λ α ε L ∧
      (∀ o, ¬ (1 + α) / 2 < (π (g o) ω₀).real (near ε (Aopt o ω₀)) →
        (1 - α) / 2 ≤ (π (g o) ω₀).real (far ε (Aopt o ω₀))) ∧
      (∀ o o', o' ≠ o → AnsweredWell π g Aopt α ε o →
        (1 + α) / 2 < (π (g o) ω₀).real (near ε (Aopt o ω₀)))

end Madre
