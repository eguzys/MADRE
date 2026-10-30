import MadreProof.Proofs.WeightedStrongDuality
import MadreProof.Proofs.Instance

/-! The weighted airtime bracket at the simulator's constants, the unit-weight theorems
re-derived from the weighted ones, and the weighted explicit-edge LP. -/

open Finset

namespace Madre

theorem airtimeBracketW : AirtimeBracketW := by
  intro V C D _ _ _ _ _ P L nbr hL hR hN hself hc hD
  have hfit := fun i => chargeFitsRadio L hL nbr hN hself i
  have hRpos : 0 < P.R := by rw [hR]; norm_num [simR]
  refine ⟨lpOptimumExistsW P _ hc hD (fun i => (hfit i).2.trans hR.ge),
    lpOptimumExistsW P 0 hc hD (fun i => by simpa using hRpos.le), ?_⟩
  intro τ τ0 μ ν lam ρ hτ hτ0 hopt
  refine ⟨?_, propositionAW P (onAirBytes L) nbr τ τ0 μ ν lam ρ hτ hτ0 hopt⟩
  obtain ⟨hb0, hb⟩ := onAirBytes_bounds L hL
  have hK : ∀ i, 0 ≤ charge (onAirBytes L) nbr i ∧
      charge (onAirBytes L) nbr i ≤ 8 * simN * onAirBytes L := by
    intro i
    refine ⟨(hfit i).1, ?_⟩
    have hc7 : ((nbr i).card : ℝ) ≤ 7 := by exact_mod_cast card_nbr_le nbr hN hself i
    unfold charge simN
    push_cast
    nlinarith
  have hmem := witnessScalingW P (charge (onAirBytes L) nbr) (8 * simN * onAirBytes L) τ0 hRpos
    hK (by unfold simN; positivity) (by rw [hR]; unfold simN simR; push_cast; nlinarith)
    hτ0.1
  have := hτ.2 hmem
  unfold floorFactor
  rw [← hR]
  linarith


section FromWeighted

variable {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]

theorem P1.values_toW (P : P1 V C D) (κ : D → ℝ) : P.values κ = P.toW.values κ := by
  ext x
  simp only [P1.values, P1W.values, Set.mem_ofPred_eq, ((unitWeightSpecialCase P).1 κ)]

theorem P1.dualOptimal_toW (P : P1 V C D) (κ : D → ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) : P.DualOptimal κ μ ν lam ρ ↔ P.toW.DualOptimal κ μ ν lam ρ := by
  obtain ⟨-, hd, ho⟩ := unitWeightSpecialCase P
  simp only [P1.DualOptimal, P1W.DualOptimal, hd, ho]

end FromWeighted

/-- `PropositionA` (unit weights) as a corollary of `PropositionAW`. -/
theorem propositionA_fromWeighted : PropositionA := by
  intro V C D _ _ _ _ P b nbr τ τ0 μ ν lam ρ hτ hτ0 hopt
  rw [P.values_toW] at hτ hτ0
  exact propositionAW P.toW b nbr τ τ0 μ ν lam ρ hτ hτ0 ((P.dualOptimal_toW 0 μ ν lam ρ).mp hopt)

/-- `LPStrongDuality` (unit weights) as a corollary of `LPStrongDualityW`. -/
theorem lpStrongDuality_fromWeighted : LPStrongDuality := by
  intro V C D _ _ _ _ P κ τ0 hτ0
  rw [P.values_toW] at hτ0
  obtain ⟨μ, ν, lam, ρ, hopt, hval⟩ := lpStrongDualityW P.toW κ τ0 hτ0
  exact ⟨μ, ν, lam, ρ, (P.dualOptimal_toW κ μ ν lam ρ).mpr hopt,
    ((unitWeightSpecialCase P).2.2 κ μ lam ρ).trans hval⟩

/-- `ComplementarySlackness` (unit weights) as a corollary of `ComplementarySlacknessW`. -/
theorem complementarySlackness_fromWeighted : ComplementarySlackness := by
  intro V C D _ _ _ _ P κ f τ μ ν lam ρ hf hτ hopt i hslack
  rw [P.values_toW] at hτ
  refine complementarySlacknessW P.toW κ f τ μ ν lam ρ (((unitWeightSpecialCase P).1 κ f τ).mp hf)
    hτ ((P.dualOptimal_toW κ μ ν lam ρ).mp hopt) i ?_
  rw [P.radio_toW]; exact hslack

/-- `AirtimeBracket` (unit weights) as a corollary of `AirtimeBracketW`. -/
theorem airtimeBracket_fromWeighted : AirtimeBracket := by
  intro V C D _ _ _ _ _ P L nbr hL hR hN hself hc hD
  have h := airtimeBracketW P.toW L nbr hL hR hN hself hc hD
  simp only [← P.values_toW] at h
  refine ⟨h.1, h.2.1, fun τ τ0 μ ν lam ρ hτ hτ0 hopt => ?_⟩
  exact h.2.2 τ τ0 μ ν lam ρ hτ hτ0 ((P.dualOptimal_toW 0 μ ν lam ρ).mp hopt)

end Madre
