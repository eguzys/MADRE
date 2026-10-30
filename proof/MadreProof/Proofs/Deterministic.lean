import MadreProof.Statement.Deterministic
import MadreProof.Proofs.Information

/-! Proof of Proposition 1(b) for a deterministic receiver, with α = 1. -/

open MeasureTheory Set

namespace Madre

/-- Moves within ε of optimal-move sets that are 3ε apart are MORE than ε apart. -/
theorem near_sep_strict {X : Type} [MetricSpace X] {ε : ℝ} {A A' : Set X}
    (h : ∀ a ∈ A, ∀ a' ∈ A', 3 * ε ≤ dist a a') :
    ∀ x ∈ near ε A, ∀ y ∈ near ε A', ε < dist x y := by
  rintro x ⟨a, ha, hxa⟩ y ⟨a', ha', hya⟩
  have h3 := h a ha a' ha'
  have t1 := dist_triangle a x a'
  have t2 := dist_triangle x y a'
  rw [dist_comm] at hxa
  linarith

theorem propositionBDeterministic : PropositionBDeterministic := by
  intro X Ω O _ _ _ _ L β Λ ε hβ hΛ hε φ hLip Aopt hsep g hg
  classical
  have := Fintype.ofFinite O
  constructor
  · have hset : {o | DetAnsweredWell φ g Aopt ε o}
        = ↑(Finset.univ.filter (fun o => DetAnsweredWell φ g Aopt ε o)) := by
      ext o; simp
    rw [hset, Set.ncard_coe_finset]
    refine card_le_cellCount _ hβ hΛ one_pos hε g (fun o _ => hg o) ?_
    intro o ho o' ho' hne hd
    obtain ⟨ω, hω⟩ := hsep o o' hne
    have hω' : Separated Aopt ε o' o ω := fun a ha a' ha' => by
      rw [dist_comm]; exact hω a' ha' a ha
    have g1 := (Finset.mem_filter.mp ho).2 ω ⟨o', Ne.symm hne, hω⟩
    have g2 := (Finset.mem_filter.mp ho').2 ω ⟨o, hne, hω'⟩
    have hgt := near_sep_strict hω _ g1 _ g2
    have hle := hLip _ (hg o) _ (hg o') ω
    linarith
  · intro o ho
    simp only [DetAnsweredWell, not_forall, exists_prop] at ho
    obtain ⟨ω, hω, hnot⟩ := ho
    have hfar : φ (g o) ω ∈ far ε (Aopt o ω) := by rw [far_eq_compl]; exact hnot
    refine ⟨ω, hω, hfar, ?_⟩
    rw [measureReal_def, Measure.dirac_apply_of_mem hfar]
    simp

theorem diracAnsweredWellIff : DiracAnsweredWellIff := by
  intro X Ω O _ _ _ L φ g Aopt α ε hα1 hα2 o
  unfold AnsweredWell DetAnsweredWell
  refine forall_congr' fun ω => imp_congr_right fun _ => ?_
  rw [measureReal_def, Measure.dirac_apply' _ (isOpen_near ε _).measurableSet]
  by_cases h : φ (g o) ω ∈ near ε (Aopt o ω)
  · simp only [h, indicator_of_mem, Pi.one_apply, ENNReal.toReal_one, iff_true]
    linarith
  · simp only [h, not_false_eq_true, indicator_of_notMem, ENNReal.toReal_zero, iff_false,
      not_lt]
    linarith

end Madre
