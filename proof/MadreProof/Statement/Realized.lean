/-
# The REALIZED move of the trained head satisfies the hypothesis of Proposition 1(b)

Statements only (proofs in `MadreProof/Proofs/Realized.lean`). `Statement/Receivers.lean`
models the commanded move. Two more maps act before the drone actually moves, and both are
included here:
* torchrl's `SafeTanhTransform` clamps tanh to [-c, c] with c = 1 − (float resolution)
  (`torchrl/modules/distributions/utils.py`, `_SafeTanhNoEps.forward`);
* INET's `handleIfOutside(REFLECT, …)` (`RLControlledMobility.cc` line 64) reflects the new
  position at the arena walls, coordinate by coordinate: x ↦ 2·lo − x below lo, 2·hi − x above hi.
The realized displacement from the current position p is reflect(p + commanded move) − p.
**Assumption:** the receiver state ω carries the drone's current position (`pos ω`), which it
does (the observation contains the drone's position). The Lipschitz constant
Λ = 60 (Λ_loc + 2 Λ_σ) survives both maps, and the deterministic evaluated head (ξ = 0) is
60 Λ_loc-Lipschitz.
-/
import MadreProof.Statement.Receivers

open MeasureTheory

namespace Madre

/-- One-coordinate wall reflection into [lo, hi] (a single reflection, as in INET). -/
noncomputable def reflect1 (lo hi x : ℝ) : ℝ :=
  if x < lo then 2 * lo - x else if hi < x then 2 * hi - x else x

/-- torchrl's safe tanh: tanh clamped to [-c, c]. -/
noncomputable def safeTanh (c a : ℝ) : ℝ := max (-c) (min c (Real.tanh a))

/-- The commanded move: 60 · safeTanh(loc + σ ⊙ ξ), clipped to the 60 m disk. -/
noncomputable def commandedMove (c : ℝ) (loc σ ξ : Fin 2 → ℝ) : Plane :=
  diskClip 60 (WithLp.toLp 2 (fun j => 60 * safeTanh c (loc j + σ j * ξ j)))

/-- The realized displacement from position p in the arena [lo, hi]². -/
noncomputable def realizedMove (c lo hi : ℝ) (p loc σ ξ : Fin 2 → ℝ) : Plane :=
  WithLp.toLp 2 (fun j => reflect1 lo hi (p j + commandedMove c loc σ ξ j) - p j)

/-- The law of the realized displacement: pushforward of the noise law ν. -/
noncomputable def realizedLaw (ν : Measure (Fin 2 → ℝ)) (c lo hi : ℝ) (p loc σ : Fin 2 → ℝ) :
    Measure Plane :=
  ν.map (realizedMove c lo hi p loc σ)

/-- **The realized move satisfies the hypotheses of Proposition 1(b)** with standard Gaussian
noise and Λ = 60 (Λ_loc + 2 Λ_σ); the current position is read from ω. -/
def RealizedSatisfiesB : Prop :=
  ∀ {Ω : Type} {L : ℕ} (β Λloc Λσ c lo hi : ℝ) (loc σ : (Fin L → ℝ) → Ω → (Fin 2 → ℝ))
    (pos : Ω → Fin 2 → ℝ), lo ≤ hi → 0 ≤ Λloc → 0 ≤ Λσ →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω, msgDist (loc m ω) (loc m' ω) ≤ Λloc * msgDist m m') →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω, msgDist (σ m ω) (σ m' ω) ≤ Λσ * msgDist m m') →
    (∀ m ω, IsProbabilityMeasure (realizedLaw stdGauss2 c lo hi (pos ω) (loc m ω) (σ m ω))) ∧
    ∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      W1 (realizedLaw stdGauss2 c lo hi (pos ω) (loc m ω) (σ m ω))
        (realizedLaw stdGauss2 c lo hi (pos ω) (loc m' ω) (σ m' ω))
        ≤ ENNReal.ofReal (60 * (Λloc + 2 * Λσ) * msgDist m m')

/-- **The deterministic evaluated head** (noise ξ = 0) moves by a 60 Λ_loc-Lipschitz function
of the message, so `PropositionBDeterministic` applies to it with Λ = 60 Λ_loc. -/
def RealizedDeterministicLipschitz : Prop :=
  ∀ {Ω : Type} {L : ℕ} (β Λloc c lo hi : ℝ) (loc σ : (Fin L → ℝ) → Ω → (Fin 2 → ℝ))
    (pos : Ω → Fin 2 → ℝ), lo ≤ hi →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω, msgDist (loc m ω) (loc m' ω) ≤ Λloc * msgDist m m') →
    ∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      dist (realizedMove c lo hi (pos ω) (loc m ω) (σ m ω) 0)
        (realizedMove c lo hi (pos ω) (loc m' ω) (σ m' ω) 0) ≤ 60 * Λloc * msgDist m m'

end Madre
