/-
# Theory note, Lemma lem:pdl and Theorem thm:floor (ii), (iii): formal statement

Statements only, no proofs (they are in `MadreProof/Proofs/Throughput.lean`).

**Setting, and how it is narrower than the note.** All randomness lives on a FINITE
probability space `Ξ` (outcomes ξ with weights `p ξ`), and the simulator state space `S`
is FINITE. The note allows a general probability space. The encoder is DETERMINISTIC,
m_t = g(o_t), as g_θ in the paper; the note also allows a randomized encoder in (iii).
**Consequence for the receiver:** on a finite `Ξ` the hypothesis `hrecv` of
`TheoremFloorIII` forces every receiver law h(·|m_t, ω_t) at a visited (m, ω) to be finitely
supported (proved as `ReceiverFiniteSupport` below). So Theorem 1(iii) is checked here for
discrete-move and deterministic receivers only (the evaluated MADRE policy is deterministic,
`evaluation_deterministic_actions: true`), NOT for Gaussian heads, which the note's Section 1
discusses (they are covered by the general-probability-space version in
`Statement/ThroughputGeneral.lean`). Joint moves `𝔸`, receiver moves `X` (a metric space) and receiver states `Ω` (a
measurable space) are arbitrary types, and the policy of every drone may be randomized and
history dependent (it is whatever produced the process).
Time: Lean step `t = 0, …, T-1` is the note's step `t + 1`, and `V T` is the note's
V*_{T+1} ≡ 0.
-/
import MadreProof.Statement

open MeasureTheory
open scoped ENNReal

namespace Madre

/-- A finite probability space: weights `p ξ ≥ 0` summing to one. -/
structure FinProb (Ξ : Type) [Fintype Ξ] where
  p : Ξ → ℝ
  nonneg : ∀ ξ, 0 ≤ p ξ
  sum_one : ∑ ξ, p ξ = 1

variable {Ξ : Type} [Fintype Ξ]

/-- Expectation 𝔼[Y]. -/
def FinProb.E (P : FinProb Ξ) (Y : Ξ → ℝ) : ℝ := ∑ ξ, P.p ξ * Y ξ

open Classical in
/-- Probability ℙ(A). -/
noncomputable def FinProb.Pr (P : FinProb Ξ) (A : Ξ → Prop) : ℝ :=
  P.E (fun ξ => if A ξ then 1 else 0)

/-- The swarm as a controlled process over `T` steps: states `s t`, joint moves `a t`,
uncharged throughput `r(s, a)` (the note's r = τ(0) at the positions `a` produces from
`s`), and a transition law `Ptr s a s'`. `markov` is the only property of the
dynamics used: 𝔼[φ(s_{t+1})] = 𝔼[∑_{s'} Ptr(s_t, a_t, s') φ(s')], a consequence of the
note's Markov property (law of s_{t+1} given the past depends only on (s_t, a_t)). -/
structure Process (Ξ S 𝔸 : Type) [Fintype Ξ] [Fintype S] where
  P : FinProb Ξ
  T : ℕ
  s : ℕ → Ξ → S
  a : ℕ → Ξ → 𝔸
  r : S → 𝔸 → ℝ
  Ptr : S → 𝔸 → S → ℝ
  markov : ∀ t < T, ∀ φ : S → ℝ,
    P.E (fun ξ => φ (s (t + 1) ξ)) = P.E (fun ξ => ∑ s', Ptr (s t ξ) (a t ξ) s' * φ s')

variable {S 𝔸 : Type} [Fintype S]

/-- The full-information optimal values Q*_t, V*_t of the note, by their defining
equations: Q*_t(s,a) = r(s,a) + ∑ Ptr(s,a,s') V*_{t+1}(s'), V*_t(s) = sup_a Q*_t(s,a)
(the max when attained), V*_T = 0. -/
structure OptValues (M : Process Ξ S 𝔸) where
  V : ℕ → S → ℝ
  Q : ℕ → S → 𝔸 → ℝ
  bellmanQ : ∀ t < M.T, ∀ s a, Q t s a = M.r s a + ∑ s', M.Ptr s a s' * V (t + 1) s'
  bellmanV : ∀ t < M.T, ∀ s, IsLUB (Set.range (Q t s)) (V t s)
  terminal : ∀ s, V M.T s = 0

/-- gap_t = V*_t(s_t) − Q*_t(s_t, a_t). -/
def OptValues.gap {M : Process Ξ S 𝔸} (O : OptValues M) (t : ℕ) (ξ : Ξ) : ℝ :=
  O.V t (M.s t ξ) - O.Q t (M.s t ξ) (M.a t ξ)

/-- **Lemma lem:pdl (performance difference).** For every process (every joint policy,
randomized or history dependent), ∑_t 𝔼[r(s_t,a_t)] = 𝔼[V*_1(s_1)] − ∑_t 𝔼[gap_t], and
every 𝔼[gap_t] ≥ 0. -/
def PerformanceDifference : Prop :=
  ∀ {Ξ S 𝔸 : Type} [Fintype Ξ] [Fintype S] (M : Process Ξ S 𝔸) (O : OptValues M),
    ∑ t ∈ Finset.range M.T, M.P.E (fun ξ => M.r (M.s t ξ) (M.a t ξ))
        = M.P.E (fun ξ => O.V 0 (M.s 0 ξ)) - ∑ t ∈ Finset.range M.T, M.P.E (O.gap t) ∧
      ∀ t < M.T, 0 ≤ M.P.E (O.gap t)

/-- The receiver's optimal moves A*_t(s): receiver components `recv a` of the joint moves
`a` with Q*_t(s, a) = V*_t(s). -/
def OptValues.Astar {X : Type} {M : Process Ξ S 𝔸} (O : OptValues M) (recv : 𝔸 → X)
    (t : ℕ) (s : S) : Set X :=
  {x | ∃ a, recv a = x ∧ O.Q t s a = O.V t s}

/-- **Theorem thm:floor (ii) (counting, integrated against ν).** `F` is an ε-separated
family: for distinct o, o' ∈ F the optimal-move sets `Aopt o ω`, `Aopt o' ω` are 3ε apart
for every ω in the measurable set `Ωsep o o'`, and ν(Ωsep o o') ≥ 1. The miss probability
q(o, ω, m) = h(far_ε(A*(o, ω)) | m, ω) is measurable in ω (the note's assumption).
Then for every assignment of messages m_o ∈ [-β,β]^L,
∑_{o∈F} ∫ q(o, ω, m_o) ν(dω) ≥ (1 − α)/2 · (|F| − C_α(L))⁺. -/
def TheoremFloorII : Prop :=
  ∀ {X Ω O : Type} [MetricSpace X] [MeasurableSpace X] [BorelSpace X] [MeasurableSpace Ω]
    (L : ℕ) (β Λ α ε : ℝ), 0 < β → 0 < Λ → 0 < α → α < 1 → 0 < ε →
    ∀ (h : (Fin L → ℝ) → Ω → Measure X), (∀ m ω, IsProbabilityMeasure (h m ω)) →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      W1 (h m ω) (h m' ω) ≤ ENNReal.ofReal (Λ * msgDist m m')) →
    ∀ (Aopt : O → Ω → Set X) (F : Finset O) (Ωsep : O → O → Set Ω) (ν : Measure Ω),
    (∀ o o', MeasurableSet (Ωsep o o')) →
    (∀ o ∈ F, ∀ o' ∈ F, o ≠ o' → ∀ ω ∈ Ωsep o o', Separated Aopt ε o o' ω) →
    (∀ o ∈ F, ∀ o' ∈ F, o ≠ o' → 1 ≤ ν (Ωsep o o')) →
    (∀ o m, Measurable fun ω => h m ω (far ε (Aopt o ω))) →
    ∀ (m : O → Fin L → ℝ), (∀ o ∈ F, m o ∈ box L β) →
      ENNReal.ofReal ((1 - α) / 2 * max ((F.card : ℝ) - cellCount β Λ α ε L) 0)
        ≤ ∑ o ∈ F, ∫⁻ ω, h (m o) ω (far ε (Aopt o ω)) ∂ν

/-- **Theorem thm:floor (iii) (throughput loss).**

* `M`, `Opt`: the process and its full-information values (above).
* `recv a`: the receiver drone j's component of the joint move `a`.
* `obs s`: the sender's private observation o(s); `ω t ξ`: the receiver state ω_t.
* `g`: the encoder, m_t = g(o_t), into [-β,β]^L. `h m ω`: the receiver kernel, a
  probability measure, W₁-Λ-Lipschitz in m on [-β,β]^L.
* `hrecv`: given (s_t, ω_t) (hence m_t), the receiver's move has law h(·|m_t, ω_t):
  ℙ(recv(a_t) ∈ B(s_t, ω_t)) = 𝔼[h(B(s_t, ω_t) | m_t, ω_t)] for every measurable-set-valued B.
* `Aopt o ω` ⊇ A*_t(s_t) whenever o_t = o, ω_t = ω (the note's union A*(o, ω) qualifies).
* `F`, `Ωsep`, `ν`: the ε-separated family and the visiting hypothesis (Assumption
  ass:visit): occ({o} × B) ≥ w ν(B) for o ∈ F and measurable B, ν(Ωsep o o') ≥ 1, where
  occ is the occupancy law (1/T) ∑_t ℙ((o_t, ω_t) ∈ ·).
* `ℓ`: any 0 ≤ ℓ ≤ the note's ℓ(ε) (= the inf of V* − Q* over moves ε off A*_t(s), o(s) ∈ F).
* `thr t ξ`: the charged throughput τ_{t+1}(κ) produced by move t, at most r(s_t, a_t).

Conclusion: τ̄ ≤ τ_full − (1 − α)/2 · ℓ · w · (|F| − C_α(L))⁺ with
τ̄ = (1/T) ∑_t 𝔼[thr_t] and τ_full = (1/T) 𝔼[V*_1(s_1)]. -/
def TheoremFloorIII : Prop :=
  ∀ {Ξ S 𝔸 X O Ω : Type} [Fintype Ξ] [Fintype S]
    [MetricSpace X] [MeasurableSpace X] [BorelSpace X] [MeasurableSpace Ω]
    (M : Process Ξ S 𝔸) (Opt : OptValues M) (recv : 𝔸 → X) (obs : S → O) (ω : ℕ → Ξ → Ω)
    (L : ℕ) (β Λ α ε : ℝ), 0 < β → 0 < Λ → 0 < α → α < 1 → 0 < ε →
    ∀ (g : O → Fin L → ℝ), (∀ o, g o ∈ box L β) →
    ∀ (h : (Fin L → ℝ) → Ω → Measure X), (∀ m ω, IsProbabilityMeasure (h m ω)) →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      W1 (h m ω) (h m' ω) ≤ ENNReal.ofReal (Λ * msgDist m m')) →
    (∀ t < M.T, ∀ B : S → Ω → Set X, (∀ s ω', MeasurableSet (B s ω')) →
      M.P.Pr (fun ξ => recv (M.a t ξ) ∈ B (M.s t ξ) (ω t ξ))
        = M.P.E (fun ξ => (h (g (obs (M.s t ξ))) (ω t ξ)).real (B (M.s t ξ) (ω t ξ)))) →
    ∀ (Aopt : O → Ω → Set X),
    (∀ t < M.T, ∀ ξ, Opt.Astar recv t (M.s t ξ) ⊆ Aopt (obs (M.s t ξ)) (ω t ξ)) →
    (∀ o m, Measurable fun ω' => h m ω' (far ε (Aopt o ω'))) →
    ∀ (F : Finset O) (Ωsep : O → O → Set Ω),
    (∀ o o', MeasurableSet (Ωsep o o')) →
    (∀ o ∈ F, ∀ o' ∈ F, o ≠ o' → ∀ ω' ∈ Ωsep o o', Separated Aopt ε o o' ω') →
    ∀ (w : ℝ) (ν : Measure Ω), 0 < w →
    (∀ o ∈ F, ∀ B, MeasurableSet B →
      ENNReal.ofReal w * ν B ≤ ENNReal.ofReal
        ((1 / M.T) * ∑ t ∈ Finset.range M.T, M.P.Pr (fun ξ => obs (M.s t ξ) = o ∧ ω t ξ ∈ B))) →
    (∀ o ∈ F, ∀ o' ∈ F, o ≠ o' → 1 ≤ ν (Ωsep o o')) →
    ∀ (ℓ : ℝ), 0 ≤ ℓ →
    (∀ t < M.T, ∀ s a, obs s ∈ F → recv a ∈ far ε (Opt.Astar recv t s) →
      ℓ ≤ Opt.V t s - Opt.Q t s a) →
    ∀ (thr : ℕ → Ξ → ℝ), (∀ t < M.T, ∀ ξ, thr t ξ ≤ M.r (M.s t ξ) (M.a t ξ)) →
      (1 / M.T) * ∑ t ∈ Finset.range M.T, M.P.E (thr t)
        ≤ (1 / M.T) * M.P.E (fun ξ => Opt.V 0 (M.s 0 ξ))
          - (1 - α) / 2 * ℓ * w * max ((F.card : ℝ) - cellCount β Λ α ε L) 0

/-- **Finite support of the receiver laws** (a consequence of `hrecv`, not a hypothesis):
on the finite space `Ξ`, the receiver laws at visited points put, on average, no mass
outside the finitely many moves recv(a_t(ξ)) that occur. -/
def ReceiverFiniteSupport : Prop :=
  ∀ {Ξ S 𝔸 X O Ω : Type} [Fintype Ξ] [Fintype S] [MetricSpace X] [MeasurableSpace X]
    [BorelSpace X] (M : Process Ξ S 𝔸) (recv : 𝔸 → X) (obs : S → O) (ω : ℕ → Ξ → Ω) {L : ℕ}
    (g : O → Fin L → ℝ) (h : (Fin L → ℝ) → Ω → Measure X),
    (∀ t < M.T, ∀ B : S → Ω → Set X, (∀ s ω', MeasurableSet (B s ω')) →
      M.P.Pr (fun ξ => recv (M.a t ξ) ∈ B (M.s t ξ) (ω t ξ))
        = M.P.E (fun ξ => (h (g (obs (M.s t ξ))) (ω t ξ)).real (B (M.s t ξ) (ω t ξ)))) →
    ∀ t < M.T, M.P.E (fun ξ => (h (g (obs (M.s t ξ))) (ω t ξ)).real
      (Set.range (fun ξ' => recv (M.a t ξ')))ᶜ) = 0

/-- **Attainment is needed (hand-check).** If at some step t < T and some state s with o(s) ∈ F
the Bellman maximum is NOT attained (no joint move a with Q*_t(s, a) = V*_t(s)), then A*_t(s)
is empty, every move is ε-far from it, and the hypothesis on ℓ in `TheoremFloorIII` forces
ℓ ≤ 0, so the bound says nothing. The note assumes the maxima are attained. -/
def NonAttainmentForcesEllZero : Prop :=
  ∀ {Ξ S 𝔸 X O : Type} [Fintype Ξ] [Fintype S] [MetricSpace X]
    (M : Process Ξ S 𝔸) (Opt : OptValues M) (recv : 𝔸 → X) (obs : S → O) (F : Finset O)
    (ε ℓ : ℝ),
    (∀ t < M.T, ∀ s a, obs s ∈ F → recv a ∈ far ε (Opt.Astar recv t s) →
      ℓ ≤ Opt.V t s - Opt.Q t s a) →
    (∃ t < M.T, ∃ s, obs s ∈ F ∧ ∀ a, Opt.Q t s a ≠ Opt.V t s) → ℓ ≤ 0

end Madre
