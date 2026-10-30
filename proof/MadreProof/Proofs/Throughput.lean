import MadreProof.Statement.Throughput
import MadreProof.Proofs.Information

/-! Proofs of Lemma lem:pdl and Theorem thm:floor (ii), (iii) of the theory note. -/

open MeasureTheory Set Finset
open scoped ENNReal

namespace Madre

section Expectation

variable {Ξ : Type} [Fintype Ξ] (P : FinProb Ξ)

theorem FinProb.E_add (Y Z : Ξ → ℝ) : P.E (fun ξ => Y ξ + Z ξ) = P.E Y + P.E Z := by
  simp only [FinProb.E, mul_add, sum_add_distrib]

theorem FinProb.E_sub (Y Z : Ξ → ℝ) : P.E (fun ξ => Y ξ - Z ξ) = P.E Y - P.E Z := by
  simp only [FinProb.E, mul_sub, sum_sub_distrib]

theorem FinProb.E_const_mul (c : ℝ) (Y : Ξ → ℝ) : P.E (fun ξ => c * Y ξ) = c * P.E Y := by
  simp only [FinProb.E, mul_sum]
  exact sum_congr rfl fun ξ _ => by ring

theorem FinProb.E_sum {ι : Type} (s : Finset ι) (Y : ι → Ξ → ℝ) :
    P.E (fun ξ => ∑ i ∈ s, Y i ξ) = ∑ i ∈ s, P.E (Y i) := by
  simp only [FinProb.E, mul_sum]
  exact sum_comm

theorem FinProb.E_mono {Y Z : Ξ → ℝ} (h : ∀ ξ, Y ξ ≤ Z ξ) : P.E Y ≤ P.E Z :=
  sum_le_sum fun ξ _ => mul_le_mul_of_nonneg_left (h ξ) (P.nonneg ξ)

theorem FinProb.E_nonneg {Y : Ξ → ℝ} (h : ∀ ξ, 0 ≤ Y ξ) : 0 ≤ P.E Y :=
  sum_nonneg fun ξ _ => mul_nonneg (P.nonneg ξ) (h ξ)

end Expectation

theorem OptValues.gap_nonneg {Ξ S 𝔸 : Type} [Fintype Ξ] [Fintype S] {M : Process Ξ S 𝔸}
    (O : OptValues M) {t : ℕ} (ht : t < M.T) (ξ : Ξ) : 0 ≤ O.gap t ξ := by
  unfold OptValues.gap
  have := (O.bellmanV t ht (M.s t ξ)).1 ⟨M.a t ξ, rfl⟩
  linarith

theorem performanceDifference : PerformanceDifference := by
  intro Ξ S 𝔸 _ _ M O
  refine ⟨?_, fun t ht => M.P.E_nonneg (O.gap_nonneg ht)⟩
  set f : ℕ → ℝ := fun t => M.P.E (fun ξ => O.V t (M.s t ξ)) with hf
  have hstep : ∀ t ∈ range M.T, M.P.E (fun ξ => M.r (M.s t ξ) (M.a t ξ))
      = -(f (t + 1) - f t) - M.P.E (O.gap t) := by
    intro t ht
    have ht' := mem_range.mp ht
    have hQ : M.P.E (fun ξ => O.Q t (M.s t ξ) (M.a t ξ))
        = M.P.E (fun ξ => M.r (M.s t ξ) (M.a t ξ))
          + M.P.E (fun ξ => ∑ s', M.Ptr (M.s t ξ) (M.a t ξ) s' * O.V (t + 1) s') := by
      rw [← FinProb.E_add]
      congr 1; funext ξ; exact O.bellmanQ t ht' _ _
    have hM := M.markov t ht' (O.V (t + 1))
    have hg : M.P.E (O.gap t) = f t - M.P.E (fun ξ => O.Q t (M.s t ξ) (M.a t ξ)) := by
      rw [hf]; simp only; rw [← FinProb.E_sub]; rfl
    rw [hf] at hg ⊢
    simp only at hg ⊢
    linarith
  rw [sum_congr rfl hstep, sum_sub_distrib, sum_neg_distrib, sum_range_sub]
  have hT : f M.T = 0 := by simp [hf, FinProb.E, O.terminal]
  rw [hT]
  simp [hf]

/-! ### Theorem 1 (ii) -/

/-- Pointwise pair inequality (Lemma lem:pair in its sum form): two laws within αε in W₁
cannot both stay within ε of optimal moves 3ε apart with total mass above 1 + α. -/
theorem far_pair {X Ω : Type} [MetricSpace X] [MeasurableSpace X] [BorelSpace X] {L : ℕ}
    {α ε : ℝ} (hα : 0 ≤ α) (hε : 0 < ε) (h : (Fin L → ℝ) → Ω → Measure X)
    (hπ : ∀ m ω, IsProbabilityMeasure (h m ω)) {m m' : Fin L → ℝ} {ω : Ω}
    (hW : W1 (h m ω) (h m' ω) ≤ ENNReal.ofReal (α * ε)) {A A' : Set X}
    (hsep : ∀ a ∈ A, ∀ a' ∈ A', 3 * ε ≤ dist a a') :
    ENNReal.ofReal (1 - α) ≤ h m ω (far ε A) + h m' ω (far ε A') := by
  have := hπ m ω
  have := hπ m' ω
  rcases lt_or_ge (1 + α) ((h m ω).real (near ε A) + (h m' ω).real (near ε A')) with hlt | hge
  · exact absurd (lt_of_lt_of_le (W1_gt_of_far_sets_sum _ _ (isOpen_near ε A).measurableSet
      (isOpen_near ε A').measurableSet hε hα (near_sep hsep) hlt) hW) (lt_irrefl _)
  · have e1 : (h m ω).real (far ε A) = 1 - (h m ω).real (near ε A) := by
      rw [far_eq_compl, measureReal_compl (isOpen_near ε A).measurableSet, probReal_univ]
    have e2 : (h m' ω).real (far ε A') = 1 - (h m' ω).real (near ε A') := by
      rw [far_eq_compl, measureReal_compl (isOpen_near ε A').measurableSet, probReal_univ]
    rw [← ofReal_measureReal (μ := h m ω) (s := far ε A),
      ← ofReal_measureReal (μ := h m' ω) (s := far ε A'),
      ← ENNReal.ofReal_add measureReal_nonneg measureReal_nonneg]
    exact ENNReal.ofReal_le_ofReal (by linarith)

theorem theoremFloorII : TheoremFloorII := by
  intro X Ω O _ _ _ _ L β Λ α ε hβ hΛ hα0 hα1 hε h hπ hLip Aopt F Ωsep ν hΩm hsep hν hmeas
    m hm
  classical
  set q : O → Ω → ℝ≥0∞ := fun o ω => h (m o) ω (far ε (Aopt o ω)) with hq
  set Qo : O → ℝ≥0∞ := fun o => ∫⁻ ω, q o ω ∂ν with hQo
  have hγ0 : 0 ≤ 1 - α := by linarith
  have pairI : ∀ o ∈ F, ∀ o' ∈ F, o ≠ o' → Λ * msgDist (m o) (m o') ≤ α * ε →
      ENNReal.ofReal (1 - α) ≤ Qo o + Qo o' := by
    intro o ho o' ho' hne hd
    have hpt : ∀ ω ∈ Ωsep o o', ENNReal.ofReal (1 - α) ≤ q o ω + q o' ω := fun ω hω =>
      far_pair hα0.le hε h hπ
        ((hLip _ (hm o ho) _ (hm o' ho') ω).trans (ENNReal.ofReal_le_ofReal hd))
        (hsep o ho o' ho' hne ω hω)
    calc ENNReal.ofReal (1 - α) = ENNReal.ofReal (1 - α) * 1 := (mul_one _).symm
      _ ≤ ENNReal.ofReal (1 - α) * ν (Ωsep o o') := by gcongr; exact hν o ho o' ho' hne
      _ = ∫⁻ ω, (Ωsep o o').indicator (fun _ => ENNReal.ofReal (1 - α)) ω ∂ν :=
          (lintegral_indicator_const (hΩm o o') _).symm
      _ ≤ ∫⁻ ω, (q o ω + q o' ω) ∂ν := by
          apply lintegral_mono
          intro ω
          by_cases hω : ω ∈ Ωsep o o'
          · rw [indicator_of_mem hω]; exact hpt ω hω
          · rw [indicator_of_notMem hω]; exact zero_le
      _ = Qo o + Qo o' := lintegral_add_left (hmeas o (m o)) _
  set Bad := F.filter (fun o => Qo o < ENNReal.ofReal ((1 - α) / 2)) with hBad
  have hBadC : Bad.card ≤ cellCount β Λ α ε L := by
    refine card_le_cellCount Bad hβ hΛ hα0 hε m (fun o ho => hm o (mem_filter.mp ho).1) ?_
    intro o ho o' ho' hne hd
    have h1 := pairI o (mem_filter.mp ho).1 o' (mem_filter.mp ho').1 hne hd
    have h2 : Qo o + Qo o' < ENNReal.ofReal (1 - α) := by
      calc Qo o + Qo o'
          < ENNReal.ofReal ((1 - α) / 2) + ENNReal.ofReal ((1 - α) / 2) :=
            ENNReal.add_lt_add (mem_filter.mp ho).2 (mem_filter.mp ho').2
        _ = ENNReal.ofReal (1 - α) := by
            rw [← ENNReal.ofReal_add (by linarith) (by linarith)]; ring_nf
    exact absurd (lt_of_le_of_lt h1 h2) (lt_irrefl _)
  have hsub : Bad ⊆ F := filter_subset _ _
  have hgood : ∀ o ∈ F \ Bad, ENNReal.ofReal ((1 - α) / 2) ≤ Qo o := by
    intro o ho
    rw [Finset.mem_sdiff] at ho
    by_contra hlt
    exact ho.2 (mem_filter.mpr ⟨ho.1, not_le.mp hlt⟩)
  have hcard : max ((F.card : ℝ) - cellCount β Λ α ε L) 0 ≤ ((F \ Bad).card : ℝ) := by
    rw [card_sdiff_of_subset hsub, Nat.cast_sub (Finset.card_le_card hsub)]
    have : (Bad.card : ℝ) ≤ cellCount β Λ α ε L := by exact_mod_cast hBadC
    have : (Bad.card : ℝ) ≤ F.card := by exact_mod_cast Finset.card_le_card hsub
    rcases le_total (F.card : ℝ) (cellCount β Λ α ε L) with hle | hle
    · rw [max_eq_right (by linarith)]; linarith
    · rw [max_eq_left (by linarith)]; linarith
  calc ENNReal.ofReal ((1 - α) / 2 * max ((F.card : ℝ) - cellCount β Λ α ε L) 0)
      = ENNReal.ofReal (max ((F.card : ℝ) - cellCount β Λ α ε L) 0)
          * ENNReal.ofReal ((1 - α) / 2) := by
        rw [mul_comm, ENNReal.ofReal_mul (le_max_right _ _)]
    _ ≤ ((F \ Bad).card : ℝ≥0∞) * ENNReal.ofReal ((1 - α) / 2) := by
        gcongr
        rw [← ENNReal.ofReal_natCast]
        exact ENNReal.ofReal_le_ofReal hcard
    _ = ∑ o ∈ F \ Bad, ENNReal.ofReal ((1 - α) / 2) := by rw [sum_const, nsmul_eq_mul]
    _ ≤ ∑ o ∈ F \ Bad, Qo o := sum_le_sum hgood
    _ ≤ ∑ o ∈ F, Qo o := sum_le_sum_of_subset sdiff_subset

/-! ### Theorem 1 (iii) -/

theorem ofReal_sum_mul_toReal {I : Type} [Fintype I] (c : I → ℝ) (x : I → ℝ≥0∞)
    (hc : ∀ i, 0 ≤ c i) (hx : ∀ i, x i ≠ ⊤) :
    ENNReal.ofReal (∑ i, c i * (x i).toReal) = ∑ i, ENNReal.ofReal (c i) * x i := by
  rw [ENNReal.ofReal_sum_of_nonneg (fun i _ => mul_nonneg (hc i) ENNReal.toReal_nonneg)]
  refine sum_congr rfl fun i _ => ?_
  rw [ENNReal.ofReal_mul (hc i), ENNReal.ofReal_toReal (hx i)]

open Classical in
/-- The occupancy step: the visiting hypothesis lifts from sets to the miss probability. -/
theorem occupancy_step {Ξ S Ω O : Type} [Fintype Ξ] [MeasurableSpace Ω]
    (P : FinProb Ξ) (T : ℕ) (st : ℕ → Ξ → S) (obs : S → O) (ω : ℕ → Ξ → Ω)
    (o : O) (q : Ω → ℝ≥0∞) (hq : Measurable q) (hq1 : ∀ x, q x ≠ ⊤)
    (w : ℝ) (ν : Measure Ω)
    (hvisit : ∀ B, MeasurableSet B →
      ENNReal.ofReal w * ν B ≤ ENNReal.ofReal
        ((1 / T) * ∑ t ∈ range T, P.Pr (fun ξ => obs (st t ξ) = o ∧ ω t ξ ∈ B))) :
    ENNReal.ofReal w * ∫⁻ x, q x ∂ν ≤ ENNReal.ofReal ((1 / T) * ∑ t ∈ range T,
      P.E (fun ξ => if obs (st t ξ) = o then (q (ω t ξ)).toReal else 0)) := by
  classical
  set c : Fin T × Ξ → ℝ := fun i => P.p i.2 / T * (if obs (st i.1 i.2) = o then 1 else 0)
    with hc
  have hc0 : ∀ i, 0 ≤ c i := fun i => by
    simp only [hc]
    have := P.nonneg i.2
    split_ifs <;> positivity
  set μ : Measure Ω := Measure.sum (fun i : Fin T × Ξ =>
    ENNReal.ofReal (c i) • Measure.dirac (ω i.1 i.2)) with hμ
  -- rewriting a time-and-outcome sum as a sum over `Fin T × Ξ`
  have hre : ∀ (Y : Ξ → Ω → ℝ),
      (1 / (T : ℝ)) * ∑ t ∈ range T, P.E (fun ξ =>
          if obs (st t ξ) = o then Y ξ (ω t ξ) else 0)
        = ∑ i : Fin T × Ξ, c i * Y i.2 (ω i.1 i.2) := by
    intro Y
    rw [Finset.sum_range, Fintype.sum_prod_type, mul_sum]
    refine sum_congr rfl fun t _ => ?_
    simp only [FinProb.E, mul_sum, hc]
    refine sum_congr rfl fun ξ _ => ?_
    split_ifs <;> ring
  have hocc : ∀ B, MeasurableSet B → μ B = ENNReal.ofReal
      ((1 / T) * ∑ t ∈ range T, P.Pr (fun ξ => obs (st t ξ) = o ∧ ω t ξ ∈ B)) := by
    intro B hB
    have e : (1 / (T : ℝ)) * ∑ t ∈ range T, P.Pr (fun ξ => obs (st t ξ) = o ∧ ω t ξ ∈ B)
        = ∑ i : Fin T × Ξ, c i * (B.indicator (fun _ => (1 : ℝ≥0∞)) (ω i.1 i.2)).toReal := by
      rw [← hre (fun _ x => (B.indicator (fun _ => (1 : ℝ≥0∞)) x).toReal)]
      congr 1
      refine sum_congr rfl fun t _ => ?_
      simp only [FinProb.Pr, FinProb.E]
      refine sum_congr rfl fun ξ _ => ?_
      by_cases h1 : obs (st t ξ) = o <;> by_cases h2 : ω t ξ ∈ B <;> simp [h1, h2]
    rw [e, ofReal_sum_mul_toReal c _ hc0 (fun i => by
      by_cases h : ω i.1 i.2 ∈ B <;> simp [h])]
    rw [hμ, Measure.sum_apply _ hB, tsum_fintype]
    refine sum_congr rfl fun i _ => ?_
    rw [Measure.smul_apply, Measure.dirac_apply' _ hB, smul_eq_mul]
    rfl
  have hle : ENNReal.ofReal w • ν ≤ μ := Measure.le_iff.mpr fun B hB => by
    rw [Measure.smul_apply, smul_eq_mul, hocc B hB]
    exact hvisit B hB
  have hint : ∫⁻ x, q x ∂μ = ∑ i : Fin T × Ξ, ENNReal.ofReal (c i) * q (ω i.1 i.2) := by
    rw [hμ, lintegral_sum_measure, tsum_fintype]
    refine sum_congr rfl fun i _ => ?_
    rw [lintegral_smul_measure, lintegral_dirac' _ hq, smul_eq_mul]
  calc ENNReal.ofReal w * ∫⁻ x, q x ∂ν = ∫⁻ x, q x ∂(ENNReal.ofReal w • ν) := by
        rw [lintegral_smul_measure, smul_eq_mul]
    _ ≤ ∫⁻ x, q x ∂μ := lintegral_mono' hle le_rfl
    _ = _ := by
        rw [hint, hre (fun _ x => (q x).toReal), ofReal_sum_mul_toReal c _ hc0 (fun i => hq1 _)]

theorem theoremFloorIII : TheoremFloorIII := by
  intro Ξ S 𝔸 X O Ω _ _ _ _ _ _ M Opt recv obs ω L β Λ α ε hβ hΛ hα0 hα1 hε g hg h hπ hLip
    hrecv Aopt hA hmeasq F Ωsep hΩm hsep w ν hw hvisit hν ℓ hℓ0 hℓ thr hthr
  classical
  set q : O → Ω → ℝ≥0∞ := fun o ω' => h (g o) ω' (far ε (Aopt o ω')) with hq
  have hq1 : ∀ o x, q o x ≠ ⊤ := fun o x => by
    have := hπ (g o) x
    exact measure_ne_top _ _
  set B : S → Ω → Set X := fun s ω' => if obs s ∈ F then far ε (Aopt (obs s) ω') else ∅
    with hB
  have hBm : ∀ s ω', MeasurableSet (B s ω') := by
    intro s ω'
    simp only [hB]
    split_ifs
    · rw [far_eq_compl]; exact (isOpen_near ε _).measurableSet.compl
    · exact MeasurableSet.empty
  set Y : O → ℕ → Ξ → ℝ := fun o t ξ =>
    if obs (M.s t ξ) = o then (q o (ω t ξ)).toReal else 0 with hY
  -- Step A: one step of the gap
  have stepA : ∀ t < M.T, ℓ * ∑ o ∈ F, M.P.E (Y o t) ≤ M.P.E (Opt.gap t) := by
    intro t ht
    have hpt : ∀ ξ, ℓ * (if recv (M.a t ξ) ∈ B (M.s t ξ) (ω t ξ) then 1 else 0)
        ≤ Opt.gap t ξ := by
      intro ξ
      split_ifs with hin
      · simp only [hB] at hin
        split_ifs at hin with hF
        · rw [mul_one]
          exact hℓ t ht _ _ hF (fun a' ha' => hin a' (hA t ht ξ ha'))
        · exact absurd hin (notMem_empty _)
      · rw [mul_zero]; exact Opt.gap_nonneg ht ξ
    have h1 : ℓ * M.P.Pr (fun ξ => recv (M.a t ξ) ∈ B (M.s t ξ) (ω t ξ))
        ≤ M.P.E (Opt.gap t) := by
      unfold FinProb.Pr
      rw [← FinProb.E_const_mul]
      exact M.P.E_mono hpt
    rw [hrecv t ht B hBm] at h1
    have hΨ : (fun ξ => (h (g (obs (M.s t ξ))) (ω t ξ)).real (B (M.s t ξ) (ω t ξ)))
        = fun ξ => ∑ o ∈ F, Y o t ξ := by
      funext ξ
      simp only [hY, hB, hq]
      rw [sum_ite_eq]
      split_ifs <;> simp [measureReal_def]
    rw [hΨ, FinProb.E_sum] at h1
    exact h1
  -- Step B and C: occupancy and counting
  set R : O → ℝ := fun o => (1 / M.T) * ∑ t ∈ range M.T, M.P.E (Y o t) with hR
  have hR0 : ∀ o, 0 ≤ R o := fun o => by
    simp only [hR]
    refine mul_nonneg (by positivity) (sum_nonneg fun t _ => M.P.E_nonneg fun ξ => ?_)
    simp only [hY]; split_ifs <;> simp
  have stepB : ∀ o ∈ F, ENNReal.ofReal w * ∫⁻ x, q o x ∂ν ≤ ENNReal.ofReal (R o) :=
    fun o ho => occupancy_step M.P M.T M.s obs ω o (q o) (hmeasq o (g o)) (hq1 o) w ν
      (hvisit o ho)
  have stepC := theoremFloorII L β Λ α ε hβ hΛ hα0 hα1 hε h hπ hLip Aopt F Ωsep ν hΩm hsep
    hν hmeasq g (fun o _ => hg o)
  have hBC : w * ((1 - α) / 2 * max ((F.card : ℝ) - cellCount β Λ α ε L) 0)
      ≤ ∑ o ∈ F, R o := by
    rw [← ENNReal.ofReal_le_ofReal_iff (sum_nonneg fun o _ => hR0 o),
      ENNReal.ofReal_mul hw.le, ENNReal.ofReal_sum_of_nonneg (fun o _ => hR0 o)]
    calc ENNReal.ofReal w * ENNReal.ofReal ((1 - α) / 2
            * max ((F.card : ℝ) - cellCount β Λ α ε L) 0)
        ≤ ENNReal.ofReal w * ∑ o ∈ F, ∫⁻ x, q o x ∂ν := by gcongr
      _ = ∑ o ∈ F, ENNReal.ofReal w * ∫⁻ x, q o x ∂ν := by rw [mul_sum]
      _ ≤ ∑ o ∈ F, ENNReal.ofReal (R o) := sum_le_sum stepB
  -- assembly with the performance-difference lemma
  obtain ⟨hpdl, -⟩ := performanceDifference M Opt
  have hT0 : 0 ≤ 1 / (M.T : ℝ) := by positivity
  have hthrsum : ∑ t ∈ range M.T, M.P.E (thr t)
      ≤ ∑ t ∈ range M.T, M.P.E (fun ξ => M.r (M.s t ξ) (M.a t ξ)) :=
    sum_le_sum fun t ht => M.P.E_mono (hthr t (mem_range.mp ht))
  have hgapsum : ℓ * ∑ o ∈ F, ∑ t ∈ range M.T, M.P.E (Y o t)
      ≤ ∑ t ∈ range M.T, M.P.E (Opt.gap t) := by
    rw [sum_comm, mul_sum]
    exact sum_le_sum fun t ht => stepA t (mem_range.mp ht)
  have e : ℓ * ∑ o ∈ F, R o = 1 / (M.T : ℝ) * (ℓ * ∑ o ∈ F, ∑ t ∈ range M.T, M.P.E (Y o t)) := by
    simp only [hR, ← mul_sum]; ring
  have a1 := mul_le_mul_of_nonneg_left hthrsum hT0
  have a2 := mul_le_mul_of_nonneg_left hgapsum hT0
  have a3 := mul_le_mul_of_nonneg_left hBC hℓ0
  have e2 : (1 - α) / 2 * ℓ * w * max ((F.card : ℝ) - cellCount β Λ α ε L) 0
      = ℓ * (w * ((1 - α) / 2 * max ((F.card : ℝ) - cellCount β Λ α ε L) 0)) := by ring
  rw [hpdl] at a1
  rw [e2]
  have e3 : 1 / (M.T : ℝ) * (M.P.E (fun ξ => Opt.V 0 (M.s 0 ξ))
      - ∑ t ∈ range M.T, M.P.E (Opt.gap t)) = 1 / (M.T : ℝ) * M.P.E (fun ξ => Opt.V 0 (M.s 0 ξ))
      - 1 / (M.T : ℝ) * ∑ t ∈ range M.T, M.P.E (Opt.gap t) := by ring
  linarith

/-- Argument of the second independent review (Oct 17). -/
theorem receiverFiniteSupport : ReceiverFiniteSupport := by
  intro Ξ S 𝔸 X O Ω _ _ _ _ _ M recv obs ω L g h hrecv t ht
  classical
  have hm : MeasurableSet (Set.range (fun ξ' => recv (M.a t ξ')))ᶜ := by
    have : (Set.range (fun ξ' => recv (M.a t ξ'))) = ⋃ ξ', {recv (M.a t ξ')} := by
      ext x; simp
    rw [this]
    exact (MeasurableSet.iUnion fun ξ' => measurableSet_singleton _).compl
  rw [← hrecv t ht (fun _ _ => _) (fun _ _ => hm)]
  simp [FinProb.Pr, FinProb.E]

/-- If ℓ ≤ V − Q(a) for every a and V is the supremum of Q, then ℓ ≤ 0. -/
theorem le_zero_of_le_sub_lub {𝔸 : Type} (Qf : 𝔸 → ℝ) (Vv ℓ : ℝ)
    (hlub : IsLUB (Set.range Qf) Vv) (h : ∀ a, ℓ ≤ Vv - Qf a) : ℓ ≤ 0 := by
  by_contra hpos
  push Not at hpos
  have hub : Vv - ℓ ∈ upperBounds (Set.range Qf) := by
    rintro _ ⟨a, rfl⟩; linarith [h a]
  have := hlub.2 hub
  linarith

theorem nonAttainmentForcesEllZero : NonAttainmentForcesEllZero := by
  intro Ξ S 𝔸 X O _ _ _ M Opt recv obs F ε ℓ hℓ ⟨t, ht, s, hF, hna⟩
  have hempty : Opt.Astar recv t s = ∅ := by
    ext x
    simp only [OptValues.Astar, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨a, -, ha⟩
    exact hna a ha
  refine le_zero_of_le_sub_lub (Opt.Q t s) (Opt.V t s) ℓ (Opt.bellmanV t ht s) fun a => ?_
  refine hℓ t ht s a hF ?_
  rw [hempty]
  intro a' ha'
  exact absurd ha' (Set.notMem_empty a')

end Madre
