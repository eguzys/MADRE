/-
# Theory note, Lemma lem:pdl and Theorem thm:floor (iii) on a GENERAL probability space

Statements only (proofs in `MadreProof/Proofs/ThroughputGeneral.lean`). This is the
measure-theoretic version of `Statement/Throughput.lean`: the outcome space Ξ carries any
probability measure P, the state space S, the joint-move space 𝔸, the receiver-move space X
and the receiver-state space Ω are measurable spaces of any size, and the receiver law
h(·|m, ω) is any probability measure (Gaussian, tanh-Normal, discrete, Dirac, ...). The
encoder is deterministic, m_t = g(o_t), as g_θ in the paper.

Hypotheses beyond the finite version, all standard side conditions:
* s_t, a_t, ω_t are measurable maps of the outcome, `recv` is measurable, and each level set
  {s | o(s) = o} of the sender observation is measurable;
* integrability of V*_t(s_t), Q*_t(s_t, a_t), r(s_t, a_t) and of the charged throughput;
* the note's measurability assumption on {(ω, x) : dist(x, A*(o, ω)) ≥ ε} (`hfar`) and on
  the miss probability ω ↦ q(o, ω, m) (`hmeasq`);
* 0 < T.
Time: Lean step `t = 0, …, T-1` is the note's step `t + 1`; `V T` is the note's V*_{T+1} ≡ 0.
-/
import MadreProof.Statement.Throughput

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace Madre

/-- The swarm as a controlled process on a general probability space: states `s t`, joint
moves `a t`, uncharged throughput `r(s, a)`. -/
structure MProcess (Ξ S 𝔸 : Type) [MeasurableSpace Ξ] where
  P : Measure Ξ
  isProb : IsProbabilityMeasure P
  T : ℕ
  s : ℕ → Ξ → S
  a : ℕ → Ξ → 𝔸
  r : S → 𝔸 → ℝ

variable {Ξ S 𝔸 : Type} [MeasurableSpace Ξ]

/-- Full-information values Q*_t, V*_t and the next-step value NV_t(s, a), which stands for
𝔼[V*_{t+1}(s_{t+1}) | s_t = s, a_t = a] = ∫ V*_{t+1} dP(·|s, a) for the transition kernel.
* `bellmanQ`, `bellmanV`, `terminal`: the note's Bellman equations, V* = sup_a Q*.
* `markov`: 𝔼[V*_{t+1}(s_{t+1})] = 𝔼[NV_t(s_t, a_t)], what the note's Markov property gives
  (proved from a transition kernel in `markov_of_kernel`, see the proof file).
* integrability of V*_t(s_t) (t ≤ T), Q*_t(s_t, a_t) and r(s_t, a_t) (t < T). -/
structure MOptValues (M : MProcess Ξ S 𝔸) where
  V : ℕ → S → ℝ
  Q : ℕ → S → 𝔸 → ℝ
  NV : ℕ → S → 𝔸 → ℝ
  bellmanQ : ∀ t < M.T, ∀ s a, Q t s a = M.r s a + NV t s a
  bellmanV : ∀ t < M.T, ∀ s, IsLUB (Set.range (Q t s)) (V t s)
  terminal : ∀ s, V M.T s = 0
  markov : ∀ t < M.T,
    ∫ ξ, V (t + 1) (M.s (t + 1) ξ) ∂M.P = ∫ ξ, NV t (M.s t ξ) (M.a t ξ) ∂M.P
  intV : ∀ t ≤ M.T, Integrable (fun ξ => V t (M.s t ξ)) M.P
  intQ : ∀ t < M.T, Integrable (fun ξ => Q t (M.s t ξ) (M.a t ξ)) M.P
  intR : ∀ t < M.T, Integrable (fun ξ => M.r (M.s t ξ) (M.a t ξ)) M.P

/-- **The `markov` field follows from a transition kernel.** If the joint law of
((s_t, a_t), s_{t+1}) is the law of (s_t, a_t) composed with a Markov kernel κ (the note's
Markov property at step t), and V is measurable and bounded, then
𝔼[V(s_{t+1})] = 𝔼[∫ V dκ(s_t, a_t)]; i.e. `markov` holds with NV_t(s, a) = ∫ V*_{t+1} dκ(s, a). -/
def MarkovOfKernel : Prop :=
  ∀ {Ξ S 𝔸 : Type} [MeasurableSpace Ξ] [MeasurableSpace S] [MeasurableSpace 𝔸]
    (P : Measure Ξ) [IsProbabilityMeasure P] (κ : ProbabilityTheory.Kernel (S × 𝔸) S)
    [ProbabilityTheory.IsMarkovKernel κ] (st sn : Ξ → S) (at' : Ξ → 𝔸) (V : S → ℝ),
    Measurable st → Measurable at' → Measurable sn → Measurable V → (∃ C, ∀ x, |V x| ≤ C) →
    P.map (fun ξ => ((st ξ, at' ξ), sn ξ)) = (P.map fun ξ => (st ξ, at' ξ)) ⊗ₘ κ →
    ∫ ξ, V (sn ξ) ∂P = ∫ ξ, (∫ y, V y ∂(κ (st ξ, at' ξ))) ∂P

/-- gap_t = V*_t(s_t) − Q*_t(s_t, a_t). -/
def MOptValues.gap {M : MProcess Ξ S 𝔸} (O : MOptValues M) (t : ℕ) (ξ : Ξ) : ℝ :=
  O.V t (M.s t ξ) - O.Q t (M.s t ξ) (M.a t ξ)

/-- The receiver's optimal moves A*_t(s). -/
def MOptValues.Astar {X : Type} {M : MProcess Ξ S 𝔸} (O : MOptValues M) (recv : 𝔸 → X)
    (t : ℕ) (s : S) : Set X :=
  {x | ∃ a, recv a = x ∧ O.Q t s a = O.V t s}

/-- **Lemma lem:pdl, general probability space.** ∑_t 𝔼[r(s_t,a_t)] = 𝔼[V*_1(s_1)] − ∑_t 𝔼[gap_t],
and every 𝔼[gap_t] ≥ 0. -/
def PerformanceDifferenceGeneral : Prop :=
  ∀ {Ξ S 𝔸 : Type} [MeasurableSpace Ξ] (M : MProcess Ξ S 𝔸) (O : MOptValues M),
    ∑ t ∈ Finset.range M.T, ∫ ξ, M.r (M.s t ξ) (M.a t ξ) ∂M.P
        = ∫ ξ, O.V 0 (M.s 0 ξ) ∂M.P - ∑ t ∈ Finset.range M.T, ∫ ξ, O.gap t ξ ∂M.P ∧
      ∀ t < M.T, 0 ≤ ∫ ξ, O.gap t ξ ∂M.P

/-- **Theorem thm:floor (iii), general probability space.** Same objects and conclusion as
`TheoremFloorIII`, with these differences:
* `hrecv` is the note's receiver condition as a joint law: for every measurable
  B ⊆ S × Ω × X, ℙ((s_t, ω_t, recv(a_t)) ∈ B) = 𝔼[h(B_{(s_t, ω_t)} | g(o_t), ω_t)], where
  B_{(s, ω)} = {x | (s, ω, x) ∈ B}; i.e. the law of (s_t, ω_t, a_t(j)) is the law of
  (s_t, ω_t) composed with the receiver kernel;
* the visiting hypothesis is Assumption ass:visit with ℙ the measure P;
* the measurability and integrability side conditions listed in the file header. -/
def TheoremFloorIIIGeneral : Prop :=
  ∀ {Ξ S 𝔸 X O Ω : Type} [MeasurableSpace Ξ] [MeasurableSpace S] [MeasurableSpace 𝔸]
    [MetricSpace X] [MeasurableSpace X] [BorelSpace X] [MeasurableSpace Ω]
    (M : MProcess Ξ S 𝔸) (Opt : MOptValues M) (recv : 𝔸 → X) (obs : S → O) (ω : ℕ → Ξ → Ω)
    (L : ℕ) (β Λ α ε : ℝ), 0 < β → 0 < Λ → 0 < α → α < 1 → 0 < ε → 0 < M.T →
    (∀ t, Measurable (M.s t)) → (∀ t, Measurable (M.a t)) → (∀ t, Measurable (ω t)) →
    Measurable recv → (∀ o, MeasurableSet {s | obs s = o}) →
    ∀ (g : O → Fin L → ℝ), (∀ o, g o ∈ box L β) →
    ∀ (h : (Fin L → ℝ) → Ω → Measure X), (∀ m ω, IsProbabilityMeasure (h m ω)) →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      W1 (h m ω) (h m' ω) ≤ ENNReal.ofReal (Λ * msgDist m m')) →
    (∀ t < M.T, ∀ B : Set (S × Ω × X), MeasurableSet B →
      M.P {ξ | (M.s t ξ, ω t ξ, recv (M.a t ξ)) ∈ B}
        = ∫⁻ ξ, h (g (obs (M.s t ξ))) (ω t ξ) {x | (M.s t ξ, ω t ξ, x) ∈ B} ∂M.P) →
    ∀ (Aopt : O → Ω → Set X),
    (∀ t < M.T, ∀ ξ, Opt.Astar recv t (M.s t ξ) ⊆ Aopt (obs (M.s t ξ)) (ω t ξ)) →
    (∀ o, MeasurableSet {p : Ω × X | p.2 ∈ far ε (Aopt o p.1)}) →
    (∀ o m, Measurable fun ω' => h m ω' (far ε (Aopt o ω'))) →
    ∀ (F : Finset O) (Ωsep : O → O → Set Ω),
    (∀ o o', MeasurableSet (Ωsep o o')) →
    (∀ o ∈ F, ∀ o' ∈ F, o ≠ o' → ∀ ω' ∈ Ωsep o o', Separated Aopt ε o o' ω') →
    ∀ (w : ℝ) (ν : Measure Ω), 0 < w →
    (∀ o ∈ F, ∀ B, MeasurableSet B →
      ENNReal.ofReal w * ν B ≤ ENNReal.ofReal (1 / M.T) *
        ∑ t ∈ Finset.range M.T, M.P {ξ | obs (M.s t ξ) = o ∧ ω t ξ ∈ B}) →
    (∀ o ∈ F, ∀ o' ∈ F, o ≠ o' → 1 ≤ ν (Ωsep o o')) →
    ∀ (ℓ : ℝ), 0 ≤ ℓ →
    (∀ t < M.T, ∀ s a, obs s ∈ F → recv a ∈ far ε (Opt.Astar recv t s) →
      ℓ ≤ Opt.V t s - Opt.Q t s a) →
    ∀ (thr : ℕ → Ξ → ℝ), (∀ t < M.T, Integrable (thr t) M.P) →
    (∀ t < M.T, ∀ ξ, thr t ξ ≤ M.r (M.s t ξ) (M.a t ξ)) →
      (1 / M.T) * ∑ t ∈ Finset.range M.T, ∫ ξ, thr t ξ ∂M.P
        ≤ (1 / M.T) * ∫ ξ, Opt.V 0 (M.s 0 ξ) ∂M.P
          - (1 - α) / 2 * ℓ * w * max ((F.card : ℝ) - cellCount β Λ α ε L) 0

/-- **Attainment is needed (hand-check), general version.** As `NonAttainmentForcesEllZero`:
a non-attained Bellman maximum at an F-state forces ℓ ≤ 0 in `TheoremFloorIIIGeneral`. -/
def NonAttainmentForcesEllZeroGeneral : Prop :=
  ∀ {Ξ S 𝔸 X O : Type} [MeasurableSpace Ξ] [MetricSpace X]
    (M : MProcess Ξ S 𝔸) (Opt : MOptValues M) (recv : 𝔸 → X) (obs : S → O) (F : Finset O)
    (ε ℓ : ℝ),
    (∀ t < M.T, ∀ s a, obs s ∈ F → recv a ∈ far ε (Opt.Astar recv t s) →
      ℓ ≤ Opt.V t s - Opt.Q t s a) →
    (∃ t < M.T, ∃ s, obs s ∈ F ∧ ∀ a, Opt.Q t s a ≠ Opt.V t s) → ℓ ≤ 0

end Madre
