/-
# Why the count bound of Proposition 1(b) does not bind for the trained policies

Statements only (proofs in `MadreProof/Proofs/Threshold.lean`). This is the threshold lemma of
`notes/planning/BOUND_B_QUANT_oct17.md` (thesis workspace). Moves lie in the disk of radius
δ = 60 m, and the optimal moves of M observations are pairwise at least 3ε apart, so M is at
most the packing number of the disk (`PackingBound`). The count bound is C_1(L) = ⌈x/ε⌉^L with
x = 2βΛ√L, and x is at least the measured response range R(ω) of the receiver to the message.
If x > 2δ/3 (= 40 m) and L ≥ 4, then C_1(L) ≥ M at every ε (`CountBoundNeverBinds`), so
"at most C_1(L) of the M observations are answered well" excludes nothing. The measured R(ω)
(59.5 to 85.8 m for every trained checkpoint) is a measurement, not part of this proof.
-/
import MadreProof.Statement.Receivers
import MadreProof.Statement

namespace Madre

/-- **Packing bound.** Points of the closed disk of radius δ in the plane, pairwise at least
3ε apart, number at most (1 + 2δ/(3ε))². -/
def PackingBound : Prop :=
  ∀ (δ ε : ℝ), 0 < ε → 0 ≤ δ → ∀ (S : Finset Plane), (∀ p ∈ S, ‖p‖ ≤ δ) →
    (∀ p ∈ S, ∀ q ∈ S, p ≠ q → 3 * ε ≤ dist p q) →
    (S.card : ℝ) ≤ (1 + 2 * δ / (3 * ε)) ^ 2

/-- **Threshold lemma.** If x > 2δ/3, ε ≤ 2δ/3 and L ≥ 4, then (1 + 2δ/(3ε))² < ⌈x/ε⌉^L. -/
def ThresholdLemma : Prop :=
  ∀ (δ ε x : ℝ) (L : ℕ), 0 < ε → 0 < δ → ε ≤ 2 * δ / 3 → 2 * δ / 3 < x → 4 ≤ L →
    (1 + 2 * δ / (3 * ε)) ^ 2 < ((⌈x / ε⌉₊ ^ L : ℕ) : ℝ)

/-- **The count bound never binds.** For every ε > 0, any set of moves in the disk of radius δ
pairwise at least 3ε apart has at most ⌈x/ε⌉^L elements when x > 2δ/3 and L ≥ 4. -/
def CountBoundNeverBinds : Prop :=
  ∀ (δ ε x : ℝ) (L : ℕ), 0 < ε → 0 < δ → 2 * δ / 3 < x → 4 ≤ L →
    ∀ (S : Finset Plane), (∀ p ∈ S, ‖p‖ ≤ δ) →
    (∀ p ∈ S, ∀ q ∈ S, p ≠ q → 3 * ε ≤ dist p q) →
    S.card ≤ ⌈x / ε⌉₊ ^ L

/-- **The count bound of Proposition 1(b) excludes nothing for these policies.** Let the
receiver's response range 2βΛ√L exceed 2δ/3 (the measured response of every trained checkpoint
is 59.5–85.8 m > 40 m = 2δ/3 for δ = 60 m; a measurement, not part of the proof) and L ≥ 4.
Then for every ε > 0 and every α ∈ (0, 1], any family of M observations whose optimal-move sets
each meet the δ-disk and are pairwise 3ε-separated (`Separated`) at a common ω₀ has
M ≤ C_α(L) = `cellCount β Λ α ε L`. So "at most C_α(L) of them are answered well" is implied
by M ≤ C_α(L) and excludes nothing. -/
def CountBoundExcludesNothing : Prop :=
  ∀ {O Ω : Type} [Finite O] (L : ℕ) (β Λ α ε δ : ℝ), 0 < α → α ≤ 1 → 0 < ε → 0 < δ →
    2 * δ / 3 < 2 * β * Λ * Real.sqrt L → 4 ≤ L →
    ∀ (Aopt : O → Ω → Set Plane) (ω₀ : Ω),
    (∀ o, ∃ a ∈ Aopt o ω₀, ‖a‖ ≤ δ) →
    (∀ o o', o ≠ o' → Separated Aopt ε o o' ω₀) →
    Nat.card O ≤ cellCount β Λ α ε L

end Madre
