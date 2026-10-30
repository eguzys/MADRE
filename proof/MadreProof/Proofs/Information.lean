import MadreProof.Statement

/-! Proof of Proposition 1(b): the coupling lemma (theory note, Lemma lem:pair),
the cube covering of [-β, β]^L, and the pigeonhole step. -/

open MeasureTheory Set
open scoped ENNReal

namespace Madre

section Coupling

variable {X : Type} [MetricSpace X] [MeasurableSpace X]

/-- Coupling lemma. If two probability laws put mass above (1 + α)/2 on two measurable
sets whose points are at least ε apart, then W₁ between the laws exceeds α ε. -/
theorem W1_gt_of_far_sets_sum (P Q : Measure X) [IsProbabilityMeasure P]
    [IsProbabilityMeasure Q] {S S' : Set X} (hS : MeasurableSet S) (hS' : MeasurableSet S')
    {α ε : ℝ} (hε : 0 < ε) (hα : 0 ≤ α) (hsep : ∀ a ∈ S, ∀ a' ∈ S', ε ≤ dist a a')
    (hPQ : 1 + α < P.real S + Q.real S') :
    ENNReal.ofReal (α * ε) < W1 P Q := by
  set c : ℝ := P.real S + Q.real S' - 1 with hc
  have hcα : α < c := by rw [hc]; linarith
  have key : ∀ π ∈ couplings P Q, ENNReal.ofReal (c * ε) ≤ ∫⁻ z, edist z.1 z.2 ∂π := by
    rintro π ⟨h1, h2⟩
    have hπuniv : π univ = 1 := by
      have := congrArg (fun μ : Measure X => μ univ) h1
      rw [Measure.map_apply measurable_fst MeasurableSet.univ, preimage_univ] at this
      rw [this, measure_univ]
    have : IsProbabilityMeasure π := ⟨hπuniv⟩
    have hA : π.real (Prod.fst ⁻¹' S) = P.real S := by
      rw [← h1, measureReal_def, measureReal_def, Measure.map_apply measurable_fst hS]
    have hB : π.real (Prod.snd ⁻¹' S') = Q.real S' := by
      rw [← h2, measureReal_def, measureReal_def, Measure.map_apply measurable_snd hS']
    have hmeasB : MeasurableSet (Prod.snd ⁻¹' S' : Set (X × X)) := measurable_snd hS'
    have hui := measureReal_union_add_inter (μ := π) (s := Prod.fst ⁻¹' S) hmeasB
    have hle : π.real (Prod.fst ⁻¹' S ∪ Prod.snd ⁻¹' S') ≤ 1 := measureReal_le_one
    have hprod : (S ×ˢ S' : Set (X × X)) = Prod.fst ⁻¹' S ∩ Prod.snd ⁻¹' S' := rfl
    have hcS : c ≤ π.real (S ×ˢ S') := by rw [hprod]; linarith
    have hmeas : MeasurableSet (S ×ˢ S') := hS.prod hS'
    calc ENNReal.ofReal (c * ε) = ENNReal.ofReal ε * ENNReal.ofReal c := by
          rw [mul_comm, ENNReal.ofReal_mul hε.le]
      _ ≤ ENNReal.ofReal ε * π (S ×ˢ S') := by
          gcongr
          rw [← ofReal_measureReal]
          exact ENNReal.ofReal_le_ofReal hcS
      _ = ∫⁻ z, (S ×ˢ S').indicator (fun _ => ENNReal.ofReal ε) z ∂π :=
          (lintegral_indicator_const hmeas _).symm
      _ ≤ ∫⁻ z, edist z.1 z.2 ∂π := by
          apply lintegral_mono
          intro z
          by_cases hz : z ∈ S ×ˢ S'
          · change _ ≤ edist z.1 z.2
            rw [indicator_of_mem hz, edist_dist]
            exact ENNReal.ofReal_le_ofReal (hsep _ hz.1 _ hz.2)
          · rw [indicator_of_notMem hz]
            exact zero_le
  have hW : ENNReal.ofReal (c * ε) ≤ W1 P Q := le_iInf₂ key
  refine lt_of_lt_of_le ?_ hW
  rw [ENNReal.ofReal_lt_ofReal_iff (by nlinarith)]
  nlinarith

theorem W1_gt_of_far_sets (P Q : Measure X) [IsProbabilityMeasure P]
    [IsProbabilityMeasure Q] {S S' : Set X} (hS : MeasurableSet S) (hS' : MeasurableSet S')
    {α ε : ℝ} (hε : 0 < ε) (hα : 0 ≤ α) (hsep : ∀ a ∈ S, ∀ a' ∈ S', ε ≤ dist a a')
    (hP : (1 + α) / 2 < P.real S) (hQ : (1 + α) / 2 < Q.real S') :
    ENNReal.ofReal (α * ε) < W1 P Q :=
  W1_gt_of_far_sets_sum P Q hS hS' hε hα hsep (by linarith)

end Coupling

section Moves

variable {X : Type} [MetricSpace X]

theorem near_eq (ε : ℝ) (A : Set X) : near ε A = ⋃ a' ∈ A, Metric.ball a' ε := by
  ext a; simp [near, Metric.mem_ball]

theorem isOpen_near (ε : ℝ) (A : Set X) : IsOpen (near ε A) := by
  rw [near_eq]; exact isOpen_biUnion fun _ _ => Metric.isOpen_ball

theorem far_eq_compl (ε : ℝ) (A : Set X) : far ε A = (near ε A)ᶜ := by
  ext a; simp [near, far]

/-- Moves within ε of two sets of optimal moves that are 3ε apart are more than ε apart. -/
theorem near_sep {ε : ℝ} {A A' : Set X} (h : ∀ a ∈ A, ∀ a' ∈ A', 3 * ε ≤ dist a a') :
    ∀ x ∈ near ε A, ∀ y ∈ near ε A', ε ≤ dist x y := by
  rintro x ⟨a, ha, hxa⟩ y ⟨a', ha', hya⟩
  have h3 := h a ha a' ha'
  have t1 := dist_triangle a x a'
  have t2 := dist_triangle x y a'
  rw [dist_comm] at hxa
  linarith

end Moves

section Cubes

/-- One coordinate: two points of [0, n] with the same clamped floor are within 1. -/
theorem coord_close {x y : ℝ} {n : ℕ} (hn : 1 ≤ n) (hx0 : 0 ≤ x) (hxn : x ≤ n)
    (hy0 : 0 ≤ y) (hyn : y ≤ n) (h : min ⌊x⌋₊ (n - 1) = min ⌊y⌋₊ (n - 1)) :
    |x - y| ≤ 1 := by
  have fx := Nat.floor_le hx0
  have fx' := Nat.lt_floor_add_one x
  have fy := Nat.floor_le hy0
  have fy' := Nat.lt_floor_add_one y
  rcases lt_or_ge ⌊x⌋₊ (n - 1) with hx | hx
  · have e : ⌊y⌋₊ = ⌊x⌋₊ := by omega
    rw [e] at fy fy'
    rw [abs_le]; constructor <;> linarith
  · have hy : n - 1 ≤ ⌊y⌋₊ := by omega
    have c1 : ((n - 1 : ℕ) : ℝ) = n - 1 := by rw [Nat.cast_sub hn]; simp
    have hx' : ((n - 1 : ℕ) : ℝ) ≤ ⌊x⌋₊ := by exact_mod_cast hx
    have hy' : ((n - 1 : ℕ) : ℝ) ≤ ⌊y⌋₊ := by exact_mod_cast hy
    rw [abs_le]; constructor <;> linarith

variable {L : ℕ} {β Λ α ε : ℝ}

/-- The cube index of a message, for cubes of side 1/s in each coordinate, with
`n = ⌈2 β s⌉` cubes per coordinate (the last one closed). -/
noncomputable def cell (β s : ℝ) (n : ℕ) (m : Fin L → ℝ) : Fin L → ℕ :=
  fun j => min ⌊(m j + β) * s⌋₊ (n - 1)

/-- Two messages of [-β, β]^L in one cube of side r/√L are within Euclidean distance r. -/
theorem msgDist_le_of_cell_eq {r : ℝ} (hβ : 0 < β) (hr : 0 < r) {m m' : Fin L → ℝ}
    (hm : m ∈ box L β) (hm' : m' ∈ box L β)
    (h : cell β (Real.sqrt L / r) ⌈2 * β * (Real.sqrt L / r)⌉₊ m
      = cell β (Real.sqrt L / r) ⌈2 * β * (Real.sqrt L / r)⌉₊ m') :
    msgDist m m' ≤ r := by
  rcases Nat.eq_zero_or_pos L with hL | hL
  · subst hL
    simp [msgDist, hr.le]
  set s := Real.sqrt L / r with hs
  set n := ⌈2 * β * s⌉₊ with hn
  have hLpos : (0 : ℝ) < L := by exact_mod_cast hL
  have hsqrt : 0 < Real.sqrt L := Real.sqrt_pos.mpr hLpos
  have hs0 : 0 < s := div_pos hsqrt hr
  have hn1 : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr
    (Nat.pos_iff_ne_zero.mp (Nat.ceil_pos.mpr (by positivity)))
  have h2βs : 2 * β * s ≤ n := Nat.le_ceil _
  -- each coordinate differs by at most 1/s, i.e. |d| * √L ≤ r
  have coord : ∀ j, (m j - m' j) ^ 2 * L ≤ r ^ 2 := by
    intro j
    have hj := congrFun h j
    simp only [cell] at hj
    have a1 := abs_le.mp (hm j)
    have a2 := abs_le.mp (hm' j)
    have hx0 : 0 ≤ (m j + β) * s := mul_nonneg (by linarith) hs0.le
    have hy0 : 0 ≤ (m' j + β) * s := mul_nonneg (by linarith) hs0.le
    have hxn : (m j + β) * s ≤ n := by nlinarith
    have hyn : (m' j + β) * s ≤ n := by nlinarith
    have hc := coord_close hn1 hx0 hxn hy0 hyn hj
    have e1 : (m j + β) * s - (m' j + β) * s = (m j - m' j) * s := by ring
    rw [e1, abs_mul, abs_of_pos hs0, hs, mul_div_assoc', div_le_one hr] at hc
    have hsq : (|m j - m' j| * Real.sqrt L) ^ 2 ≤ r ^ 2 :=
      pow_le_pow_left₀ (by positivity) hc 2
    rwa [mul_pow, sq_abs, Real.sq_sqrt hLpos.le] at hsq
  have hsum : ∑ j, (m j - m' j) ^ 2 ≤ r ^ 2 := by
    have : ∀ j, (m j - m' j) ^ 2 ≤ r ^ 2 / L := fun j => by
      rw [le_div_iff₀ hLpos]; exact coord j
    calc ∑ j, (m j - m' j) ^ 2 ≤ ∑ _j : Fin L, r ^ 2 / L := Finset.sum_le_sum fun j _ => this j
      _ = r ^ 2 := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          field_simp
  unfold msgDist
  calc Real.sqrt (∑ j, (m j - m' j) ^ 2) ≤ Real.sqrt (r ^ 2) := Real.sqrt_le_sqrt hsum
    _ = r := Real.sqrt_sq hr.le

theorem cell_mem (β s : ℝ) (n : ℕ) (hn : 1 ≤ n ∨ L = 0) (m : Fin L → ℝ) :
    cell β s n m ∈ Fintype.piFinset (fun _ : Fin L => Finset.range n) := by
  rw [Fintype.mem_piFinset]
  intro j
  rcases hn with hn | hL
  · simp only [cell, Finset.mem_range]; omega
  · exact absurd j.isLt (by omega)

end Cubes

/-- Pigeonhole on the cubes: if no two distinct members of `s` may have messages within
Euclidean distance αε/Λ, then `s` has at most C_α(L) members. -/
theorem card_le_cellCount {ι : Type} (s : Finset ι) {L : ℕ} {β Λ α ε : ℝ}
    (hβ : 0 < β) (hΛ : 0 < Λ) (hα0 : 0 < α) (hε : 0 < ε)
    (g : ι → Fin L → ℝ) (hg : ∀ i ∈ s, g i ∈ box L β)
    (hpair : ∀ i ∈ s, ∀ j ∈ s, i ≠ j → Λ * msgDist (g i) (g j) ≤ α * ε → False) :
    s.card ≤ cellCount β Λ α ε L := by
  set r : ℝ := α * ε / Λ with hr
  have hr0 : 0 < r := by positivity
  set sc : ℝ := Real.sqrt L / r with hs
  set n : ℕ := ⌈2 * β * sc⌉₊ with hn
  have hcount : cellCount β Λ α ε L = n ^ L := by
    simp only [cellCount, hn, hs, hr]
    congr 2
    field_simp
  have hn1 : 1 ≤ n ∨ L = 0 := by
    rcases Nat.eq_zero_or_pos L with hL | hL
    · exact Or.inr hL
    · left
      have hLpos : (0 : ℝ) < L := by exact_mod_cast hL
      have : 0 < sc := div_pos (Real.sqrt_pos.mpr hLpos) hr0
      exact Nat.one_le_iff_ne_zero.mpr
        (Nat.pos_iff_ne_zero.mp (Nat.ceil_pos.mpr (by positivity)))
  have hT : (Fintype.piFinset fun _ : Fin L => Finset.range n).card = n ^ L := by simp
  rw [hcount, ← hT]
  refine Finset.card_le_card_of_injOn (fun i => cell β sc n (g i))
    (fun i _ => cell_mem β sc n hn1 (g i)) ?_
  intro i hi j hj hcell
  by_contra hne
  have hdist : msgDist (g i) (g j) ≤ r := msgDist_le_of_cell_eq hβ hr0 (hg i hi) (hg j hj) hcell
  refine hpair i hi j hj hne ?_
  calc Λ * msgDist (g i) (g j) ≤ Λ * r := mul_le_mul_of_nonneg_left hdist hΛ.le
    _ = α * ε := by rw [hr]; field_simp

theorem propositionB : PropositionB := by
  intro X Ω O _ _ _ _ L β Λ α ε hβ hΛ hα0 hα1 hε π hπ hLip Aopt hsep g hg
  set r : ℝ := α * ε / Λ with hr
  have hr0 : 0 < r := by positivity
  set s : ℝ := Real.sqrt L / r with hs
  set n : ℕ := ⌈2 * β * s⌉₊ with hn
  have hcount : cellCount β Λ α ε L = n ^ L := by
    simp only [cellCount, hn, hs, hr]
    congr 2
    field_simp
  have hn1 : 1 ≤ n ∨ L = 0 := by
    rcases Nat.eq_zero_or_pos L with hL | hL
    · exact Or.inr hL
    · left
      have hLpos : (0 : ℝ) < L := by exact_mod_cast hL
      have : 0 < s := div_pos (Real.sqrt_pos.mpr hLpos) hr0
      exact Nat.one_le_iff_ne_zero.mpr
        (Nat.pos_iff_ne_zero.mp (Nat.ceil_pos.mpr (by positivity)))
  constructor
  · -- pigeonhole: the cube index is injective on the observations answered well
    set T := Fintype.piFinset (fun _ : Fin L => Finset.range n)
    have hT : ((T : Set (Fin L → ℕ))).ncard = n ^ L := by
      rw [Set.ncard_coe_finset, Fintype.card_piFinset_const, Finset.card_range]
    rw [hcount, ← hT]
    refine Set.ncard_le_ncard_of_injOn (fun o => cell β s n (g o))
      (fun o _ => cell_mem β s n hn1 (g o)) ?_ (Finset.finite_toSet T)
    intro o ho o' ho' hcell
    by_contra hne
    obtain ⟨ω, hω⟩ := hsep o o' hne
    have hω' : Separated Aopt ε o' o ω := fun a ha a' ha' => by
      rw [dist_comm]; exact hω a' ha' a ha
    have g1 := ho ω ⟨o', Ne.symm hne, hω⟩
    have g2 := ho' ω ⟨o, hne, hω'⟩
    have hgt := W1_gt_of_far_sets (π (g o) ω) (π (g o') ω)
      (isOpen_near ε _).measurableSet (isOpen_near ε _).measurableSet hε hα0.le
      (near_sep hω) g1 g2
    have hdist : msgDist (g o) (g o') ≤ r := msgDist_le_of_cell_eq hβ hr0 (hg o) (hg o') hcell
    have hle := hLip (g o) (hg o) (g o') (hg o') ω
    have : ENNReal.ofReal (Λ * msgDist (g o) (g o')) ≤ ENNReal.ofReal (α * ε) := by
      apply ENNReal.ofReal_le_ofReal
      calc Λ * msgDist (g o) (g o') ≤ Λ * r := mul_le_mul_of_nonneg_left hdist hΛ.le
        _ = α * ε := by rw [hr]; field_simp
    exact absurd (lt_of_lt_of_le hgt (hle.trans this)) (lt_irrefl _)
  · -- every other observation misses with probability at least (1 - α)/2 at some such ω
    intro o ho
    simp only [AnsweredWell, not_forall, not_lt, exists_prop] at ho
    obtain ⟨ω, hω, hlow⟩ := ho
    refine ⟨ω, hω, ?_⟩
    rw [far_eq_compl, measureReal_compl (isOpen_near ε _).measurableSet, probReal_univ]
    linarith

end Madre
