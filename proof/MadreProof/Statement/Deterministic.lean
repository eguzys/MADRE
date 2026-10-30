/-
# Proposition 1(b) for a deterministic receiver (α = 1): formal statement

Statements only (proofs in `MadreProof/Proofs/Deterministic.lean`). The evaluated MADRE policy
is deterministic (`evaluation_deterministic_actions: true`): the move is a function
φ(m, ω) of the received message and the rest of the local state, i.e. its law is the Dirac
measure δ_{φ(m,ω)}. For such a receiver the count improves from C_α(L) (any α < 1) to
C_1(L) = ⌈2βΛ√L/ε⌉^L, and "lands at least ε from every optimal move" holds with probability 1.
The proof is direct (deterministic Lemma lem:pair), not a limit α → 1, so the formula holds
exactly, also when 2βΛ√L/ε is an integer.
-/
import MadreProof.Statement

open MeasureTheory

namespace Madre

/-- A deterministic receiver answers `o` well: at every such ω its move φ(g(o), ω) is
strictly within ε of an optimal move. -/
def DetAnsweredWell {X O Ω : Type} [MetricSpace X] {L : ℕ} (φ : (Fin L → ℝ) → Ω → X)
    (g : O → Fin L → ℝ) (Aopt : O → Ω → Set X) (ε : ℝ) (o : O) : Prop :=
  ∀ ω, SuchOmega Aopt ε o ω → φ (g o) ω ∈ near ε (Aopt o ω)

/-- **Proposition 1(b), deterministic receiver, α = 1.** φ is Λ-Lipschitz in m on
[-β,β]^L (this is the W₁ hypothesis for the Dirac receiver δ_{φ(m,ω)}). At most
C_1(L) = ⌈2βΛ√L/ε⌉^L observations are answered well, and every other one lands, at some
such ω, at least ε from every optimal move, with probability one. -/
def PropositionBDeterministic : Prop :=
  ∀ {X Ω O : Type} [MetricSpace X] [MeasurableSpace X] [BorelSpace X] [Finite O]
    (L : ℕ) (β Λ ε : ℝ), 0 < β → 0 < Λ → 0 < ε →
    ∀ (φ : (Fin L → ℝ) → Ω → X),
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω, dist (φ m ω) (φ m' ω) ≤ Λ * msgDist m m') →
    ∀ (Aopt : O → Ω → Set X),
    (∀ o o', o ≠ o' → ∃ ω, Separated Aopt ε o o' ω) →
    ∀ (g : O → Fin L → ℝ), (∀ o, g o ∈ box L β) →
      {o | DetAnsweredWell φ g Aopt ε o}.ncard ≤ cellCount β Λ 1 ε L ∧
      ∀ o, ¬ DetAnsweredWell φ g Aopt ε o →
        ∃ ω, SuchOmega Aopt ε o ω ∧ φ (g o) ω ∈ far ε (Aopt o ω) ∧
          (Measure.dirac (φ (g o) ω)).real (far ε (Aopt o ω)) = 1

/-- For the Dirac receiver, `AnsweredWell` of `PropositionB` at any α < 1 is exactly
`DetAnsweredWell` (probability above (1 + α)/2 of a 0/1 event means the event happens). -/
def DiracAnsweredWellIff : Prop :=
  ∀ {X Ω O : Type} [MetricSpace X] [MeasurableSpace X] [BorelSpace X] {L : ℕ}
    (φ : (Fin L → ℝ) → Ω → X) (g : O → Fin L → ℝ) (Aopt : O → Ω → Set X) (α ε : ℝ),
    -1 < α → α < 1 → ∀ o,
      AnsweredWell (fun m ω => Measure.dirac (φ m ω)) g Aopt α ε o ↔
        DetAnsweredWell φ g Aopt ε o

end Madre
