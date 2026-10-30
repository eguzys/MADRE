import MadreProof.Statement.ThroughputGeneral
import MadreProof.Proofs.Throughput

/-! Proofs of Lemma lem:pdl and Theorem thm:floor (iii) on a general probability space. -/

open MeasureTheory ProbabilityTheory Set Finset
open scoped ENNReal

namespace Madre

variable {Ξ S 𝔸 : Type} [MeasurableSpace Ξ]

theorem MOptValues.gap_nonneg {M : MProcess Ξ S 𝔸} (O : MOptValues M) {t : ℕ}
    (ht : t < M.T) (ξ : Ξ) : 0 ≤ O.gap t ξ := by
  unfold MOptValues.gap
  have := (O.bellmanV t ht (M.s t ξ)).1 ⟨M.a t ξ, rfl⟩
  linarith

theorem MOptValues.integrable_gap {M : MProcess Ξ S 𝔸} (O : MOptValues M) {t : ℕ}
    (ht : t < M.T) : Integrable (O.gap t) M.P :=
  (O.intV t ht.le).sub (O.intQ t ht)

theorem performanceDifferenceGeneral : PerformanceDifferenceGeneral := by
  intro Ξ S 𝔸 _ M O
  refine ⟨?_, fun t ht => integral_nonneg (O.gap_nonneg ht)⟩
  set f : ℕ → ℝ := fun t => ∫ ξ, O.V t (M.s t ξ) ∂M.P with hf
  have hstep : ∀ t ∈ range M.T, ∫ ξ, M.r (M.s t ξ) (M.a t ξ) ∂M.P
      = -(f (t + 1) - f t) - ∫ ξ, O.gap t ξ ∂M.P := by
    intro t ht
    have ht' := mem_range.mp ht
    have hNV : Integrable (fun ξ => O.NV t (M.s t ξ) (M.a t ξ)) M.P := by
      refine ((O.intQ t ht').sub (O.intR t ht')).congr (Filter.Eventually.of_forall fun ξ => ?_)
      simp only [O.bellmanQ t ht', Pi.sub_apply]; ring
    have hQ : ∫ ξ, O.Q t (M.s t ξ) (M.a t ξ) ∂M.P
        = ∫ ξ, M.r (M.s t ξ) (M.a t ξ) ∂M.P + ∫ ξ, O.NV t (M.s t ξ) (M.a t ξ) ∂M.P := by
      rw [← integral_add (O.intR t ht') hNV]
      congr 1; funext ξ; exact O.bellmanQ t ht' _ _
    have hM := O.markov t ht'
    have hg : ∫ ξ, O.gap t ξ ∂M.P = f t - ∫ ξ, O.Q t (M.s t ξ) (M.a t ξ) ∂M.P := by
      rw [hf]; exact integral_sub (O.intV t ht'.le) (O.intQ t ht')
    simp only [hf] at hg ⊢
    linarith
  rw [sum_congr rfl hstep, sum_sub_distrib, sum_neg_distrib, sum_range_sub]
  have hT : f M.T = 0 := by simp [hf, O.terminal]
  rw [hT]
  simp [hf]

theorem theoremFloorIIIGeneral : TheoremFloorIIIGeneral := by
  intro Ξ S 𝔸 X O Ω _ _ _ _ _ _ _ M Opt recv obs ω L β Λ α ε hβ hΛ hα0 hα1 hε hT hs ha hω
    hrecvm hobs g hg h hπ hLip hrecv Aopt hA hfar hmeasq F Ωsep hΩm hsep w ν hw hvisit hν
    ℓ hℓ0 hℓ thr hthrI hthr
  classical
  have hP := M.isProb
  set q : O → Ω → ℝ≥0∞ := fun o ω' => h (g o) ω' (far ε (Aopt o ω')) with hq
  have hqm : ∀ o, Measurable (q o) := fun o => hmeasq o (g o)
  set Bset : Set (S × Ω × X) :=
    ⋃ o ∈ F, ({s | obs s = o} ×ˢ {p : Ω × X | p.2 ∈ far ε (Aopt o p.1)}) with hBset
  have hBm : MeasurableSet Bset :=
    Finset.measurableSet_biUnion F fun o _ => (hobs o).prod (hfar o)
  have hmemB : ∀ s ω' x, (s, ω', x) ∈ Bset ↔ obs s ∈ F ∧ x ∈ far ε (Aopt (obs s) ω') := by
    intro s ω' x
    simp only [hBset, Set.mem_iUnion, Set.mem_prod, Set.mem_ofPred_eq, exists_prop]
    constructor
    · rintro ⟨o, ho, rfl, hx⟩; exact ⟨ho, hx⟩
    · rintro ⟨ho, hx⟩; exact ⟨obs s, ho, rfl, hx⟩
  set A : ℕ → O → Set Ξ := fun t o => {ξ | obs (M.s t ξ) = o} with hAdef
  have hAm : ∀ t o, MeasurableSet (A t o) := fun t o => (hs t) (hobs o)
  set Y : ℕ → O → ℝ≥0∞ := fun t o =>
    ∫⁻ ξ, (A t o).indicator (fun ξ => q o (ω t ξ)) ξ ∂M.P with hY
  -- Step A: one step of the gap, in [0, ∞]
  have stepA : ∀ t < M.T, ENNReal.ofReal ℓ * ∑ o ∈ F, Y t o
      ≤ ∫⁻ ξ, ENNReal.ofReal (Opt.gap t ξ) ∂M.P := by
    intro t ht
    set E : Set Ξ := {ξ | (M.s t ξ, ω t ξ, recv (M.a t ξ)) ∈ Bset} with hE
    have hEm : MeasurableSet E :=
      hBm.preimage ((hs t).prodMk ((hω t).prodMk (hrecvm.comp (ha t))))
    have hpt : ∀ ξ, E.indicator (fun _ => ENNReal.ofReal ℓ) ξ ≤ ENNReal.ofReal (Opt.gap t ξ) := by
      intro ξ
      by_cases hξ : ξ ∈ E
      · rw [indicator_of_mem hξ]
        obtain ⟨hF, hx⟩ := (hmemB _ _ _).mp hξ
        exact ENNReal.ofReal_le_ofReal (hℓ t ht _ _ hF (fun a' ha' => hx a' (hA t ht ξ ha')))
      · rw [indicator_of_notMem hξ]; exact zero_le
    have hsec : ∀ ξ, h (g (obs (M.s t ξ))) (ω t ξ) {x | (M.s t ξ, ω t ξ, x) ∈ Bset}
        = ∑ o ∈ F, (A t o).indicator (fun ξ => q o (ω t ξ)) ξ := by
      intro ξ
      by_cases hF : obs (M.s t ξ) ∈ F
      · have e : {x | (M.s t ξ, ω t ξ, x) ∈ Bset} = far ε (Aopt (obs (M.s t ξ)) (ω t ξ)) := by
          ext x; rw [Set.mem_ofPred_eq, hmemB]; exact ⟨fun h => h.2, fun h => ⟨hF, h⟩⟩
        rw [e, Finset.sum_eq_single (obs (M.s t ξ))]
        · rw [indicator_of_mem (show ξ ∈ A t (obs (M.s t ξ)) from rfl)]
        · intro o _ hne
          exact indicator_of_notMem (fun h' => hne (show obs (M.s t ξ) = o from h').symm) _
        · intro h'; exact absurd hF h'
      · have e : {x | (M.s t ξ, ω t ξ, x) ∈ Bset} = ∅ := by
          ext x; rw [Set.mem_ofPred_eq, hmemB]; simp [hF]
        rw [e, measure_empty]
        symm
        refine Finset.sum_eq_zero fun o ho => indicator_of_notMem ?_ _
        intro h'
        have h'' : obs (M.s t ξ) = o := h'
        exact hF (h'' ▸ ho)
    calc ENNReal.ofReal ℓ * ∑ o ∈ F, Y t o
        = ENNReal.ofReal ℓ * ∫⁻ ξ, ∑ o ∈ F, (A t o).indicator (fun ξ => q o (ω t ξ)) ξ ∂M.P := by
          rw [lintegral_finsetSum (f := fun o ξ => (A t o).indicator (fun ξ => q o (ω t ξ)) ξ) _ fun o _ => ((hqm o).comp (hω t)).indicator (hAm t o)]
      _ = ENNReal.ofReal ℓ * M.P E := by
          rw [hE, hrecv t ht Bset hBm]
          congr 1
          exact lintegral_congr fun ξ => (hsec ξ).symm
      _ = ∫⁻ ξ, E.indicator (fun _ => ENNReal.ofReal ℓ) ξ ∂M.P :=
          (lintegral_indicator_const hEm _).symm
      _ ≤ _ := lintegral_mono hpt
  -- Step B: occupancy, for each o ∈ F
  have hTr : (0 : ℝ) < M.T := by exact_mod_cast hT
  have stepB : ∀ o ∈ F, ENNReal.ofReal (M.T * w) * ∫⁻ x, q o x ∂ν
      ≤ ∑ t ∈ range M.T, Y t o := by
    intro o ho
    set μ : Measure Ω := Measure.sum fun t : Fin M.T => (M.P.restrict (A t o)).map (ω t) with hμ
    have hμB : ∀ B, MeasurableSet B →
        μ B = ∑ t ∈ range M.T, M.P {ξ | obs (M.s t ξ) = o ∧ ω t ξ ∈ B} := by
      intro B hB
      rw [hμ, Measure.sum_apply _ hB, tsum_fintype, Finset.sum_range]
      refine Finset.sum_congr rfl fun t _ => ?_
      rw [Measure.map_apply (hω t) hB, Measure.restrict_apply (hω t hB)]
      congr 1
      ext ξ; simp [hAdef, and_comm]
    have hle : ENNReal.ofReal (M.T * w) • ν ≤ μ := Measure.le_iff.mpr fun B hB => by
      rw [Measure.smul_apply, smul_eq_mul, hμB B hB, ENNReal.ofReal_mul hTr.le, mul_assoc]
      calc ENNReal.ofReal M.T * (ENNReal.ofReal w * ν B)
          ≤ ENNReal.ofReal M.T * (ENNReal.ofReal (1 / M.T) *
              ∑ t ∈ range M.T, M.P {ξ | obs (M.s t ξ) = o ∧ ω t ξ ∈ B}) := by
            gcongr ENNReal.ofReal (M.T : ℝ) * ?_
            exact hvisit o ho B hB
        _ = _ := by
            rw [← mul_assoc, ← ENNReal.ofReal_mul hTr.le, mul_one_div_cancel hTr.ne',
              ENNReal.ofReal_one, one_mul]
    have hint : ∫⁻ x, q o x ∂μ = ∑ t ∈ range M.T, Y t o := by
      rw [hμ, lintegral_sum_measure, tsum_fintype, Finset.sum_range]
      refine Finset.sum_congr rfl fun t _ => ?_
      rw [lintegral_map (hqm o) (hω t)]
      simp only [hY]
      rw [lintegral_indicator (hAm t o)]
    calc ENNReal.ofReal (M.T * w) * ∫⁻ x, q o x ∂ν
        = ∫⁻ x, q o x ∂(ENNReal.ofReal (M.T * w) • ν) := by
          rw [lintegral_smul_measure, smul_eq_mul]
      _ ≤ ∫⁻ x, q o x ∂μ := lintegral_mono' hle le_rfl
      _ = _ := hint
  -- Step C: counting
  have stepC := theoremFloorII L β Λ α ε hβ hΛ hα0 hα1 hε h hπ hLip Aopt F Ωsep ν hΩm hsep
    hν hmeasq g (fun o _ => hg o)
  -- assemble in [0, ∞]
  set G : ℝ := ∑ t ∈ range M.T, ∫ ξ, Opt.gap t ξ ∂M.P with hG
  have hG0 : 0 ≤ G := sum_nonneg fun t ht => integral_nonneg (Opt.gap_nonneg (mem_range.mp ht))
  have hmax0 : 0 ≤ max ((F.card : ℝ) - cellCount β Λ α ε L) 0 := le_max_right _ _
  have hγ : 0 ≤ (1 - α) / 2 := by linarith
  have hGbound : ℓ * (M.T * w) * ((1 - α) / 2 * max ((F.card : ℝ) - cellCount β Λ α ε L) 0)
      ≤ G := by
    rw [← ENNReal.ofReal_le_ofReal_iff hG0]
    have e1 : ENNReal.ofReal G = ∑ t ∈ range M.T, ∫⁻ ξ, ENNReal.ofReal (Opt.gap t ξ) ∂M.P := by
      rw [hG, ENNReal.ofReal_sum_of_nonneg (fun t ht =>
        integral_nonneg (Opt.gap_nonneg (mem_range.mp ht)))]
      refine Finset.sum_congr rfl fun t ht => ?_
      exact ofReal_integral_eq_lintegral_ofReal (Opt.integrable_gap (mem_range.mp ht))
        (Filter.Eventually.of_forall (Opt.gap_nonneg (mem_range.mp ht)))
    rw [e1, ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_mul hℓ0]
    calc ENNReal.ofReal ℓ * ENNReal.ofReal (M.T * w) *
          ENNReal.ofReal ((1 - α) / 2 * max ((F.card : ℝ) - cellCount β Λ α ε L) 0)
        ≤ ENNReal.ofReal ℓ * ENNReal.ofReal (M.T * w) * ∑ o ∈ F, ∫⁻ x, q o x ∂ν := by
          gcongr
      _ = ENNReal.ofReal ℓ * ∑ o ∈ F, ENNReal.ofReal (M.T * w) * ∫⁻ x, q o x ∂ν := by
          rw [mul_assoc, Finset.mul_sum]
      _ ≤ ENNReal.ofReal ℓ * ∑ o ∈ F, ∑ t ∈ range M.T, Y t o := by
          gcongr with o ho; exact stepB o ho
      _ = ∑ t ∈ range M.T, ENNReal.ofReal ℓ * ∑ o ∈ F, Y t o := by
          rw [Finset.sum_comm, Finset.mul_sum]
      _ ≤ _ := Finset.sum_le_sum fun t ht => stepA t (mem_range.mp ht)
  -- assemble with the performance-difference lemma
  obtain ⟨hpdl, -⟩ := performanceDifferenceGeneral M Opt
  have hthrsum : ∑ t ∈ range M.T, ∫ ξ, thr t ξ ∂M.P
      ≤ ∑ t ∈ range M.T, ∫ ξ, M.r (M.s t ξ) (M.a t ξ) ∂M.P :=
    Finset.sum_le_sum fun t ht => integral_mono (hthrI t (mem_range.mp ht))
      (Opt.intR t (mem_range.mp ht)) (hthr t (mem_range.mp ht))
  rw [hpdl] at hthrsum
  have hT0 : 0 ≤ 1 / (M.T : ℝ) := by positivity
  have a1 := mul_le_mul_of_nonneg_left hthrsum hT0
  have a2 := mul_le_mul_of_nonneg_left hGbound hT0
  have e2 : 1 / (M.T : ℝ) * (ℓ * (M.T * w) * ((1 - α) / 2 *
      max ((F.card : ℝ) - cellCount β Λ α ε L) 0))
      = (1 - α) / 2 * ℓ * w * max ((F.card : ℝ) - cellCount β Λ α ε L) 0 := by
    field_simp
  have e3 : 1 / (M.T : ℝ) * (∫ ξ, Opt.V 0 (M.s 0 ξ) ∂M.P - G)
      = 1 / (M.T : ℝ) * ∫ ξ, Opt.V 0 (M.s 0 ξ) ∂M.P - 1 / (M.T : ℝ) * G := by ring
  rw [hG] at e3
  linarith

theorem markovOfKernel : MarkovOfKernel := by
  intro Ξ S 𝔸 _ _ _ P _ κ _ st sn at' V hst hat hsn hV ⟨C, hC⟩ hlaw
  have hj : Measurable fun ξ => ((st ξ, at' ξ), sn ξ) := (hst.prodMk hat).prodMk hsn
  have hp : Measurable fun ξ => (st ξ, at' ξ) := hst.prodMk hat
  have hVs : StronglyMeasurable V := hV.stronglyMeasurable
  calc ∫ ξ, V (sn ξ) ∂P = ∫ p, V p.2 ∂(P.map fun ξ => ((st ξ, at' ξ), sn ξ)) := by
        exact (integral_map hj.aemeasurable
          (hVs.comp_measurable measurable_snd).aestronglyMeasurable).symm
    _ = ∫ p, V p.2 ∂((P.map fun ξ => (st ξ, at' ξ)) ⊗ₘ κ) := by rw [hlaw]
    _ = ∫ x, ∫ y, V y ∂(κ x) ∂(P.map fun ξ => (st ξ, at' ξ)) := by
        rw [Measure.integral_compProd]
        refine Integrable.of_bound (hVs.comp_measurable measurable_snd).aestronglyMeasurable C
          (Filter.Eventually.of_forall fun p => ?_)
        simpa [Real.norm_eq_abs] using hC p.2
    _ = _ := by
        exact integral_map hp.aemeasurable (hVs.integral_kernel (κ := κ)).aestronglyMeasurable

theorem nonAttainmentForcesEllZeroGeneral : NonAttainmentForcesEllZeroGeneral := by
  intro Ξ S 𝔸 X O _ _ M Opt recv obs F ε ℓ hℓ ⟨t, ht, s, hF, hna⟩
  have hempty : Opt.Astar recv t s = ∅ := by
    ext x
    simp only [MOptValues.Astar, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨a, -, ha⟩
    exact hna a ha
  refine le_zero_of_le_sub_lub (Opt.Q t s) (Opt.V t s) ℓ (Opt.bellmanV t ht s) fun a => ?_
  refine hℓ t ht s a hF ?_
  rw [hempty]
  intro a' ha'
  exact absurd ha' (Set.notMem_empty a')

end Madre
