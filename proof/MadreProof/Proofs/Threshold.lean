import MadreProof.Statement.Threshold

/-! Proofs of the packing bound, the threshold lemma, and that the count bound never binds. -/

open MeasureTheory Metric Finset

namespace Madre

theorem packingBound : PackingBound := by
  intro δ ε hε hδ S hS hsep
  set r : ℝ := 3 * ε / 2 with hr
  have hr0 : 0 < r := by positivity
  have hdisj : (S : Set Plane).PairwiseDisjoint fun p => ball p r := by
    intro p hp q hq hne
    exact ball_disjoint_ball (by have := hsep p hp q hq hne; linarith)
  have hsub : (⋃ p ∈ S, ball p r) ⊆ ball (0 : Plane) (δ + r) := by
    intro x hx
    simp only [Set.mem_iUnion, mem_ball, exists_prop] at hx
    obtain ⟨p, hp, hx⟩ := hx
    rw [mem_ball, dist_zero_right]
    have := norm_le_norm_add_norm_sub' x p
    rw [dist_eq_norm] at hx
    have h2 := hS p hp
    linarith [norm_sub_rev x p]
  have hvol := measure_mono (μ := (volume : Measure Plane)) hsub
  rw [measure_biUnion_finset hdisj (fun p _ => measurableSet_ball)] at hvol
  have hfr : Module.finrank ℝ Plane = 2 := finrank_euclideanSpace_fin
  simp only [Measure.addHaar_ball volume _ hr0.le, sum_const, nsmul_eq_mul, hfr,
    Measure.addHaar_ball volume _ (by positivity : (0 : ℝ) ≤ δ + r)] at hvol
  have hK0 : volume (ball (0 : Plane) 1) ≠ 0 := (measure_ball_pos volume _ one_pos).ne'
  have hKt : volume (ball (0 : Plane) 1) ≠ ⊤ := measure_ball_lt_top.ne
  rw [← mul_assoc, ENNReal.mul_le_mul_iff_left hK0 hKt, ← ENNReal.ofReal_natCast,
    ← ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_le_ofReal_iff (by positivity)]
    at hvol
  have e : (1 + 2 * δ / (3 * ε)) ^ 2 = (δ + r) ^ 2 / r ^ 2 := by
    rw [hr]; field_simp; ring
  rw [e, le_div_iff₀ (by positivity)]
  exact hvol

theorem thresholdLemma : ThresholdLemma := by
  intro δ ε x L hε hδ hεδ hx hL
  set y : ℝ := 2 * δ / (3 * ε) with hy
  have hy1 : 1 ≤ y := by
    rw [hy, le_div_iff₀ (by positivity)]; linarith
  have hxy : y < x / ε := by
    rw [hy, lt_div_iff₀ hε]
    have : 2 * δ / (3 * ε) * ε = 2 * δ / 3 := by field_simp
    linarith
  set b : ℕ := ⌈x / ε⌉₊ with hb
  have hbx : x / ε ≤ b := Nat.le_ceil _
  have hb2 : (2 : ℝ) ≤ b := by
    have : (1 : ℝ) < b := by linarith
    have : 1 < b := by exact_mod_cast this
    exact_mod_cast this
  have h1 : (1 + y) ^ 2 < (1 + (b : ℝ)) ^ 2 := by
    apply pow_lt_pow_left₀ (by linarith) (by linarith) (by norm_num)
  have h2 : (1 + (b : ℝ)) ^ 2 ≤ 4 * (b : ℝ) ^ 2 := by nlinarith
  have h3 : 4 * (b : ℝ) ^ 2 ≤ (b : ℝ) ^ L := by
    obtain ⟨k, rfl⟩ : ∃ k, L = k + 4 := ⟨L - 4, by omega⟩
    have hb1 : (1 : ℝ) ≤ b := by linarith
    have : (1 : ℝ) ≤ (b : ℝ) ^ k := one_le_pow₀ hb1
    have h4 : (4 : ℝ) ≤ (b : ℝ) ^ 2 := by nlinarith
    calc 4 * (b : ℝ) ^ 2 ≤ (b : ℝ) ^ 2 * (b : ℝ) ^ 2 := by nlinarith
      _ ≤ (b : ℝ) ^ k * ((b : ℝ) ^ 2 * (b : ℝ) ^ 2) := by
          have : 0 ≤ (b : ℝ) ^ 2 * (b : ℝ) ^ 2 := by positivity
          nlinarith
      _ = (b : ℝ) ^ (k + 4) := by ring
  push_cast
  linarith

theorem countBoundNeverBinds : CountBoundNeverBinds := by
  intro δ ε x L hε hδ hx hL S hS hsep
  rcases le_or_gt ε (2 * δ / 3) with hεδ | hεδ
  · have h1 := packingBound δ ε hε hδ.le S hS hsep
    have h2 := thresholdLemma δ ε x L hε hδ hεδ hx hL
    have : (S.card : ℝ) < ((⌈x / ε⌉₊ ^ L : ℕ) : ℝ) := lt_of_le_of_lt h1 h2
    exact_mod_cast this.le
  · -- 3ε > 2δ ≥ diameter: at most one point
    have hcard : S.card ≤ 1 := by
      rw [Finset.card_le_one]
      intro p hp q hq
      by_contra hne
      have h := hsep p hp q hq hne
      have hd : dist p q ≤ 2 * δ := by
        rw [dist_eq_norm]
        calc ‖p - q‖ ≤ ‖p‖ + ‖q‖ := norm_sub_le p q
          _ ≤ 2 * δ := by linarith [hS p hp, hS q hq]
      linarith
    have hpos : 0 < ⌈x / ε⌉₊ := Nat.ceil_pos.mpr (div_pos (by linarith) hε)
    calc S.card ≤ 1 := hcard
      _ ≤ ⌈x / ε⌉₊ ^ L := Nat.one_le_pow _ _ hpos

theorem countBoundExcludesNothing : CountBoundExcludesNothing := by
  intro O Ω _ L β Λ α ε δ hα0 hα1 hε hδ hx hL Aopt ω₀ hdisk hsep
  classical
  have := Fintype.ofFinite O
  choose a ha hn using hdisk
  have hinj : Function.Injective a := by
    intro o o' h
    by_contra hne
    have hd := hsep o o' hne (a o) (ha o) (a o') (ha o')
    rw [h, dist_self] at hd
    linarith
  set S : Finset Plane := Finset.univ.image a with hS
  have hcardS : S.card = Fintype.card O := by
    rw [hS, Finset.card_image_of_injective _ hinj, Finset.card_univ]
  have hbound := countBoundNeverBinds δ ε (2 * β * Λ * Real.sqrt L) L hε hδ hx hL S
    (by intro p hp; obtain ⟨o, -, rfl⟩ := Finset.mem_image.mp hp; exact hn o)
    (by
      intro p hp q hq hpq
      obtain ⟨o, -, rfl⟩ := Finset.mem_image.mp hp
      obtain ⟨o', -, rfl⟩ := Finset.mem_image.mp hq
      exact hsep o o' (fun h => hpq (h ▸ rfl)) _ (ha o) _ (ha o'))
  have hceil : ⌈2 * β * Λ * Real.sqrt L / ε⌉₊ ≤ ⌈2 * β * Λ * Real.sqrt L / (α * ε)⌉₊ := by
    apply Nat.ceil_le_ceil
    have hxpos : 0 < 2 * β * Λ * Real.sqrt L := by linarith
    rw [div_le_div_iff₀ hε (mul_pos hα0 hε)]
    have h1 : 0 ≤ 2 * β * Λ * Real.sqrt L * ε := (mul_pos hxpos hε).le
    calc 2 * β * Λ * Real.sqrt L * (α * ε) = (2 * β * Λ * Real.sqrt L * ε) * α := by ring
      _ ≤ (2 * β * Λ * Real.sqrt L * ε) * 1 := mul_le_mul_of_nonneg_left hα1 h1
      _ = 2 * β * Λ * Real.sqrt L * ε := by ring
  rw [Nat.card_eq_fintype_card, ← hcardS]
  exact hbound.trans (Nat.pow_le_pow_left hceil L)

end Madre
