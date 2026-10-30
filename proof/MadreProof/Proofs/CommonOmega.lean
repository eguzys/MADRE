import MadreProof.Proofs.Information

/-! The "one ω common to all M observations" reading of Proposition 1(b) follows from
`propositionB` by taking the receiver-state type to be a single point
(argument of the independent review, Oct 17). -/

open MeasureTheory

namespace Madre

theorem propositionB_commonOmega : PropositionBCommonOmega := by
  intro X Ω O _ _ _ _ L β Λ α ε hβ hΛ hα0 hα1 hε π hπ hLip Aopt ω₀ hsep g hg
  have h := propositionB (X := X) (Ω := Unit) (O := O) L β Λ α ε hβ hΛ hα0 hα1 hε
    (fun m _ => π m ω₀) (fun m _ => hπ m ω₀)
    (fun m hm m' hm' _ => hLip m hm m' hm' ω₀)
    (fun o _ => Aopt o ω₀) (fun o o' hne => ⟨(), hsep o o' hne⟩) g hg
  refine ⟨le_trans (Set.ncard_le_ncard ?_ (Set.toFinite _)) h.1, fun o ho => ?_,
    fun o o' hne hgood => hgood ω₀ ⟨o', hne, hsep o o' hne.symm⟩⟩
  · intro o ho _ _
    exact ho
  · have := hπ (g o) ω₀
    rw [far_eq_compl, measureReal_compl (isOpen_near ε _).measurableSet, probReal_univ]
    linarith [not_lt.mp ho]

end Madre
