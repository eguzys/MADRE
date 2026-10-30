/-
# The trained policy head satisfies the W₁-Lipschitz hypothesis of Proposition 1(b)

Statements only (proofs in `MadreProof/Proofs/Receivers.lean`). The training-time policy is
BenchMARL's MAPPO with `use_tanh_normal: True` on the action box [-1,1]² (`omnet_env.py`,
action spec `low=-1, high=1, shape=(2,)`): the action is tanh(loc + σ ⊙ ξ) componentwise with
ξ ~ N(0, I₂) and (loc, σ) produced by the network from the local state (in particular from
the received message m). `omnet_env.py` multiplies it by `OMNET_STEP_M` = 60 m, and
`RLControlledMobility.cc` clips the move radially to `maxSpeed` = 60 m/s over the 1 s step,
i.e. retracts it onto the disk of radius 60 m. The move law is therefore the pushforward of
the noise law under ξ ↦ diskClip₆₀(60 · tanh(loc + σ ⊙ ξ)).

Proved here: if m ↦ loc(m, ω) and m ↦ σ(m, ω) are Λ_loc- and Λ_σ-Lipschitz (Euclidean
norms) on [-β,β]^L, the move law is W₁-Lipschitz in m with Λ = 60 (Λ_loc + c Λ_σ), where
c = 𝔼[|ξ₀| + |ξ₁|], and c ≤ 2 for standard Gaussian noise. Λ_loc and Λ_σ are properties of
the trained network (any composition of the scale map and of the clamp `scale_lb` included);
they are NOT computed here.
-/
import MadreProof.Statement
import Mathlib.Probability.Distributions.Gaussian.Real

open MeasureTheory

namespace Madre

/-- Receiver moves: the plane with the Euclidean distance. -/
abbrev Plane := EuclideanSpace ℝ (Fin 2)

/-- The simulator's speed clip: radial retraction onto the closed disk of radius r. -/
noncomputable def diskClip (r : ℝ) (x : Plane) : Plane :=
  if ‖x‖ ≤ r then x else (r / ‖x‖) • x

/-- The move of the tanh-Normal head for network outputs (loc, σ) and noise ξ. -/
noncomputable def tanhMove (loc σ ξ : Fin 2 → ℝ) : Plane :=
  diskClip 60 (WithLp.toLp 2 (fun j => 60 * Real.tanh (loc j + σ j * ξ j)))

/-- The law of the move: pushforward of the noise law ν. -/
noncomputable def tanhNormalLaw (ν : Measure (Fin 2 → ℝ)) (loc σ : Fin 2 → ℝ) : Measure Plane :=
  ν.map (tanhMove loc σ)

/-- Standard Gaussian noise on ℝ², N(0, I₂). -/
noncomputable def stdGauss2 : Measure (Fin 2 → ℝ) :=
  Measure.pi fun _ => ProbabilityTheory.gaussianReal 0 1

/-- **W₁-Lipschitz bound for the tanh-Normal head, any noise law ν.** -/
def TanhNormalLipschitz : Prop :=
  ∀ {Ω : Type} {L : ℕ} (β Λloc Λσ : ℝ) (loc σ : (Fin L → ℝ) → Ω → (Fin 2 → ℝ))
    (ν : Measure (Fin 2 → ℝ)), IsProbabilityMeasure ν → 0 ≤ Λloc → 0 ≤ Λσ →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω, msgDist (loc m ω) (loc m' ω) ≤ Λloc * msgDist m m') →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω, msgDist (σ m ω) (σ m' ω) ≤ Λσ * msgDist m m') →
    ∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      W1 (tanhNormalLaw ν (loc m ω) (σ m ω)) (tanhNormalLaw ν (loc m' ω) (σ m' ω))
        ≤ ENNReal.ofReal (60 * Λloc * msgDist m m')
          + ENNReal.ofReal (60 * Λσ * msgDist m m') * ∫⁻ ξ, ENNReal.ofReal (|ξ 0| + |ξ 1|) ∂ν

/-- **The trained head satisfies the hypotheses of Proposition 1(b)** with standard Gaussian
noise: every move law is a probability measure, and it is W₁-Lipschitz in m on [-β,β]^L with
Λ = 60 (Λ_loc + 2 Λ_σ). -/
def TanhNormalSatisfiesB : Prop :=
  ∀ {Ω : Type} {L : ℕ} (β Λloc Λσ : ℝ) (loc σ : (Fin L → ℝ) → Ω → (Fin 2 → ℝ)),
    0 ≤ Λloc → 0 ≤ Λσ →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω, msgDist (loc m ω) (loc m' ω) ≤ Λloc * msgDist m m') →
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω, msgDist (σ m ω) (σ m' ω) ≤ Λσ * msgDist m m') →
    (∀ m ω, IsProbabilityMeasure (tanhNormalLaw stdGauss2 (loc m ω) (σ m ω))) ∧
    ∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      W1 (tanhNormalLaw stdGauss2 (loc m ω) (σ m ω)) (tanhNormalLaw stdGauss2 (loc m' ω) (σ m' ω))
        ≤ ENNReal.ofReal (60 * (Λloc + 2 * Λσ) * msgDist m m')

end Madre
