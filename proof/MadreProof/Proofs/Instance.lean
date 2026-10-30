import MadreProof.Statement.Instance
import MadreProof.Proofs.StrongDuality
import MadreProof.Proofs.NoFree

/-! Proofs for the simulator instance: κ(i) ≤ R and the two-sided airtime bracket. -/

open Finset

namespace Madre

theorem onAirBytes_bounds : ∀ L ∈ testedL, 0 ≤ onAirBytes L ∧ 64 * onAirBytes L ≤ 184448 := by
  intro L hL
  simp only [testedL, mem_insert, mem_singleton] at hL
  rcases hL with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> norm_num [onAirBytes]

theorem card_nbr_le {D : Type} [Fintype D] [DecidableEq D] (nbr : D → Finset D)
    (hN : Fintype.card D ≤ simN) (hself : ∀ i, i ∉ nbr i) (i : D) : (nbr i).card ≤ 7 := by
  have hsub : nbr i ⊆ univ.erase i := fun j hj => mem_erase.mpr ⟨fun h => hself i (h ▸ hj),
    mem_univ j⟩
  have := card_le_card hsub
  rw [card_erase_of_mem (mem_univ i), card_univ] at this
  simp only [simN] at hN
  omega

theorem chargeFitsRadio : ChargeFitsRadio := by
  intro L hL D _ _ nbr hN hself i
  obtain ⟨hb0, hb⟩ := onAirBytes_bounds L hL
  have hc : ((nbr i).card : ℝ) ≤ 7 := by exact_mod_cast card_nbr_le nbr hN hself i
  have hc0 : (0 : ℝ) ≤ (nbr i).card := Nat.cast_nonneg _
  unfold charge simR
  constructor
  · positivity
  · nlinarith

theorem airtimeBracket : AirtimeBracket := by
  intro V C D _ _ _ _ _ P L nbr hL hR hN hself hc hD
  have hfit := fun i => chargeFitsRadio L hL nbr hN hself i
  have hRpos : 0 < P.R := by rw [hR]; norm_num [simR]
  refine ⟨lpOptimumExists P _ hc hD (fun i => (hfit i).2.trans hR.ge),
    lpOptimumExists P 0 hc hD (fun i => by simpa using hRpos.le), ?_⟩
  intro τ τ0 μ ν lam ρ hτ hτ0 hopt
  refine ⟨?_, propositionA P (onAirBytes L) nbr τ τ0 μ ν lam ρ hτ hτ0 hopt⟩
  obtain ⟨hb0, hb⟩ := onAirBytes_bounds L hL
  have hK : ∀ i, 0 ≤ charge (onAirBytes L) nbr i ∧
      charge (onAirBytes L) nbr i ≤ 8 * simN * onAirBytes L := by
    intro i
    refine ⟨(hfit i).1, ?_⟩
    have hc7 : ((nbr i).card : ℝ) ≤ 7 := by exact_mod_cast card_nbr_le nbr hN hself i
    unfold charge simN
    push_cast
    nlinarith
  have hmem := witnessScaling P (charge (onAirBytes L) nbr) (8 * simN * onAirBytes L) τ0 hRpos
    hK (by unfold simN; positivity) (by rw [hR]; unfold simN simR; push_cast; nlinarith)
    hτ0.1
  have := hτ.2 hmem
  unfold floorFactor
  rw [← hR]
  linarith

theorem floorValues : FloorValues := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> norm_num [floorFactor, simN, simR, onAirBytes]

end Madre
