import MadreProof.Statement.Receivers
import MadreProof.Proofs.Information

/-! Proof that the tanh-Normal head is W₁-Lipschitz in the message (same-noise coupling). -/

open MeasureTheory Real
open scoped ENNReal

namespace Madre

theorem hasDerivAt_tanh' (x : ℝ) : HasDerivAt tanh (1 / cosh x ^ 2) x := by
  have h := (hasDerivAt_sinh x).div (hasDerivAt_cosh x) (cosh_pos x).ne'
  have e : (sinh / cosh) = tanh := by funext y; rw [Pi.div_apply, tanh_eq_sinh_div_cosh]
  rw [e] at h
  convert h using 1
  have h1 := cosh_sq x
  have h2 := (cosh_pos x).ne'
  field_simp
  nlinarith

theorem continuous_tanh' : Continuous tanh :=
  continuous_iff_continuousAt.2 fun x => (hasDerivAt_tanh' x).continuousAt

/-- tanh is 1-Lipschitz (its derivative is 1/cosh² ≤ 1). -/
theorem abs_tanh_sub_le (a b : ℝ) : |tanh a - tanh b| ≤ |a - b| := by
  have hd := hasDerivAt_tanh'
  have hL : LipschitzWith 1 tanh := lipschitzWith_of_nnnorm_deriv_le
    (fun x => (hd x).differentiableAt) (fun x => by
      rw [(hd x).deriv, ← NNReal.coe_le_coe, coe_nnnorm, Real.norm_eq_abs, NNReal.coe_one,
        abs_of_pos (by have := cosh_pos x; positivity), div_le_one (by have := cosh_pos x; positivity)]
      nlinarith [one_le_cosh x])
  simpa [Real.dist_eq] using hL.dist_le_mul a b

theorem norm_diskClip_le {r : ℝ} (hr : 0 ≤ r) (x : Plane) : ‖diskClip r x‖ ≤ r := by
  unfold diskClip
  split_ifs with h
  · exact h
  · push Not at h
    have hx : 0 < ‖x‖ := lt_of_le_of_lt hr h
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg hr hx.le), div_mul_cancel₀ _ hx.ne']

/-- Variational inequality of the radial retraction onto the disk. -/
theorem inner_diskClip_le {r : ℝ} (hr : 0 ≤ r) (x z : Plane) (hz : ‖z‖ ≤ r) :
    inner ℝ (x - diskClip r x) (z - diskClip r x) ≤ 0 := by
  unfold diskClip
  split_ifs with h
  · simp
  · push Not at h
    have hx : 0 < ‖x‖ := lt_of_le_of_lt hr h
    set c := r / ‖x‖ with hc
    have hc1 : c ≤ 1 := by rw [hc, div_le_one hx]; exact h.le
    have hcx : c * ‖x‖ = r := by rw [hc, div_mul_cancel₀ _ hx.ne']
    have e : x - c • x = (1 - c) • x := by rw [sub_smul, one_smul]
    rw [e, inner_smul_left, inner_sub_right, inner_smul_right, real_inner_self_eq_norm_sq]
    have hxz : inner ℝ x z ≤ ‖x‖ * r :=
      (real_inner_le_norm x z).trans (mul_le_mul_of_nonneg_left hz hx.le)
    simp only [conj_trivial]
    have : inner ℝ x z - c * ‖x‖ ^ 2 ≤ 0 := by
      have : c * ‖x‖ ^ 2 = r * ‖x‖ := by rw [sq, ← mul_assoc, hcx]
      rw [this]; linarith
    exact mul_nonpos_of_nonneg_of_nonpos (by linarith) this

/-- The radial retraction onto the disk is 1-Lipschitz. -/
theorem dist_diskClip_le {r : ℝ} (hr : 0 ≤ r) (x y : Plane) :
    dist (diskClip r x) (diskClip r y) ≤ dist x y := by
  set a := diskClip r x
  set b := diskClip r y
  have h1 := inner_diskClip_le hr x b (norm_diskClip_le hr y)
  have h2 := inner_diskClip_le hr y a (norm_diskClip_le hr x)
  have key : ‖a - b‖ ^ 2 ≤ inner ℝ (x - y) (a - b) := by
    have e1 : inner ℝ (x - a) (b - a) + inner ℝ (y - b) (a - b)
        = ‖a - b‖ ^ 2 - inner ℝ (x - y) (a - b) := by
      rw [← real_inner_self_eq_norm_sq]
      simp only [inner_sub_left, inner_sub_right, real_inner_comm]
      ring
    linarith
  have hcs := real_inner_le_norm (x - y) (a - b)
  rw [dist_eq_norm, dist_eq_norm]
  rcases eq_or_lt_of_le (norm_nonneg (a - b)) with h0 | hpos
  · rw [← h0]; exact norm_nonneg _
  · nlinarith

theorem sqrt_add_le2 (x0 x1 y0 y1 : ℝ) :
    Real.sqrt ((x0 + y0) ^ 2 + (x1 + y1) ^ 2)
      ≤ Real.sqrt (x0 ^ 2 + x1 ^ 2) + Real.sqrt (y0 ^ 2 + y1 ^ 2) := by
  have hA : 0 ≤ x0 ^ 2 + x1 ^ 2 := by positivity
  have hB : 0 ≤ y0 ^ 2 + y1 ^ 2 := by positivity
  have hcs : x0 * y0 + x1 * y1 ≤ Real.sqrt (x0 ^ 2 + x1 ^ 2) * Real.sqrt (y0 ^ 2 + y1 ^ 2) := by
    rw [← Real.sqrt_mul hA]
    refine le_trans (le_abs_self _) (Real.abs_le_sqrt ?_)
    nlinarith [sq_nonneg (x0 * y1 - x1 * y0)]
  rw [Real.sqrt_le_iff]
  refine ⟨by positivity, ?_⟩
  have e1 := Real.sq_sqrt hA
  have e2 := Real.sq_sqrt hB
  nlinarith

theorem sqrt_mul_le2 (b0 b1 z0 z1 : ℝ) :
    Real.sqrt ((b0 * z0) ^ 2 + (b1 * z1) ^ 2)
      ≤ Real.sqrt (b0 ^ 2 + b1 ^ 2) * (|z0| + |z1|) := by
  have hB : 0 ≤ b0 ^ 2 + b1 ^ 2 := by positivity
  rw [Real.sqrt_le_iff]
  refine ⟨by positivity, ?_⟩
  rw [show (Real.sqrt (b0 ^ 2 + b1 ^ 2) * (|z0| + |z1|)) ^ 2
      = (b0 ^ 2 + b1 ^ 2) * (|z0| + |z1|) ^ 2 by rw [mul_pow, Real.sq_sqrt hB]]
  have h0 : z0 ^ 2 ≤ (|z0| + |z1|) ^ 2 := by
    rw [← sq_abs z0]; nlinarith [abs_nonneg z0, abs_nonneg z1]
  have h1 : z1 ^ 2 ≤ (|z0| + |z1|) ^ 2 := by
    rw [← sq_abs z1]; nlinarith [abs_nonneg z0, abs_nonneg z1]
  nlinarith [mul_le_mul_of_nonneg_left h0 (sq_nonneg b0),
    mul_le_mul_of_nonneg_left h1 (sq_nonneg b1)]

theorem msgDist_two (x y : Fin 2 → ℝ) :
    msgDist x y = Real.sqrt ((x 0 - y 0) ^ 2 + (x 1 - y 1) ^ 2) := by
  simp [msgDist, Fin.sum_univ_two]

/-- Same-noise pointwise bound. -/
theorem dist_tanhMove_le (l s l' s' ξ : Fin 2 → ℝ) :
    dist (tanhMove l s ξ) (tanhMove l' s' ξ)
      ≤ 60 * msgDist l l' + 60 * msgDist s s' * (|ξ 0| + |ξ 1|) := by
  refine (dist_diskClip_le (by norm_num) _ _).trans ?_
  rw [EuclideanSpace.dist_eq, Fin.sum_univ_two]
  simp only [Real.dist_eq, sq_abs]
  set d0 := (l 0 + s 0 * ξ 0) - (l' 0 + s' 0 * ξ 0)
  set d1 := (l 1 + s 1 * ξ 1) - (l' 1 + s' 1 * ξ 1)
  have t0 : (60 * tanh (l 0 + s 0 * ξ 0) - 60 * tanh (l' 0 + s' 0 * ξ 0)) ^ 2 ≤ (60 * d0) ^ 2 := by
    rw [← sq_abs, ← sq_abs (60 * d0)]
    have := abs_tanh_sub_le (l 0 + s 0 * ξ 0) (l' 0 + s' 0 * ξ 0)
    have e : |60 * tanh (l 0 + s 0 * ξ 0) - 60 * tanh (l' 0 + s' 0 * ξ 0)|
        = 60 * |tanh (l 0 + s 0 * ξ 0) - tanh (l' 0 + s' 0 * ξ 0)| := by
      rw [← mul_sub, abs_mul]; norm_num
    rw [e, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 60)]
    exact pow_le_pow_left₀ (by positivity) (by linarith) 2
  have t1 : (60 * tanh (l 1 + s 1 * ξ 1) - 60 * tanh (l' 1 + s' 1 * ξ 1)) ^ 2 ≤ (60 * d1) ^ 2 := by
    rw [← sq_abs, ← sq_abs (60 * d1)]
    have := abs_tanh_sub_le (l 1 + s 1 * ξ 1) (l' 1 + s' 1 * ξ 1)
    have e : |60 * tanh (l 1 + s 1 * ξ 1) - 60 * tanh (l' 1 + s' 1 * ξ 1)|
        = 60 * |tanh (l 1 + s 1 * ξ 1) - tanh (l' 1 + s' 1 * ξ 1)| := by
      rw [← mul_sub, abs_mul]; norm_num
    rw [e, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 60)]
    exact pow_le_pow_left₀ (by positivity) (by linarith) 2
  have hsq : Real.sqrt ((60 * d0) ^ 2 + (60 * d1) ^ 2) = 60 * Real.sqrt (d0 ^ 2 + d1 ^ 2) := by
    rw [show (60 * d0) ^ 2 + (60 * d1) ^ 2 = 60 ^ 2 * (d0 ^ 2 + d1 ^ 2) by ring,
      Real.sqrt_mul (by norm_num), Real.sqrt_sq (by norm_num)]
  have hd : Real.sqrt (d0 ^ 2 + d1 ^ 2) ≤ msgDist l l' + msgDist s s' * (|ξ 0| + |ξ 1|) := by
    rw [msgDist_two, msgDist_two]
    have e0 : d0 = (l 0 - l' 0) + (s 0 - s' 0) * ξ 0 := by ring
    have e1 : d1 = (l 1 - l' 1) + (s 1 - s' 1) * ξ 1 := by ring
    rw [e0, e1]
    exact (sqrt_add_le2 _ _ _ _).trans (by linarith [sqrt_mul_le2 (s 0 - s' 0) (s 1 - s' 1) (ξ 0) (ξ 1)])
  calc Real.sqrt ((60 * tanh (l 0 + s 0 * ξ 0) - 60 * tanh (l' 0 + s' 0 * ξ 0)) ^ 2
        + (60 * tanh (l 1 + s 1 * ξ 1) - 60 * tanh (l' 1 + s' 1 * ξ 1)) ^ 2)
      ≤ Real.sqrt ((60 * d0) ^ 2 + (60 * d1) ^ 2) := Real.sqrt_le_sqrt (by linarith)
    _ = 60 * Real.sqrt (d0 ^ 2 + d1 ^ 2) := hsq
    _ ≤ 60 * (msgDist l l' + msgDist s s' * (|ξ 0| + |ξ 1|)) := by gcongr
    _ = _ := by ring

theorem measurable_diskClip (r : ℝ) : Measurable (diskClip r) := by
  unfold diskClip
  exact Measurable.ite (measurableSet_le measurable_norm measurable_const) measurable_id
    ((measurable_const.div measurable_norm).smul measurable_id)

theorem measurable_tanhMove (l s : Fin 2 → ℝ) : Measurable (tanhMove l s) := by
  unfold tanhMove
  refine (measurable_diskClip 60).comp ((WithLp.measurable_toLp 2 _).comp ?_)
  refine measurable_pi_iff.2 fun j => ?_
  exact measurable_const.mul (continuous_tanh'.measurable.comp
    (measurable_const.add (measurable_const.mul (measurable_pi_apply j))))

/-- Same-noise coupling: W₁ of two pushforwards is at most the mean distance. -/
theorem W1_map_le {Y Z : Type} [MetricSpace Y] [MeasurableSpace Y] [MeasurableSpace Z]
    (ν : Measure Z) {f f' : Z → Y} (hf : Measurable f) (hf' : Measurable f') :
    W1 (ν.map f) (ν.map f') ≤ ∫⁻ z, edist (f z) (f' z) ∂ν := by
  have hp : Measurable fun z => (f z, f' z) := hf.prodMk hf'
  have h1 : (ν.map fun z => (f z, f' z)).map Prod.fst = ν.map f := by
    rw [Measure.map_map measurable_fst hp]; rfl
  have h2 : (ν.map fun z => (f z, f' z)).map Prod.snd = ν.map f' := by
    rw [Measure.map_map measurable_snd hp]; rfl
  exact (iInf₂_le _ ⟨h1, h2⟩).trans (lintegral_map_le _ hp.aemeasurable)

theorem tanhNormalLipschitz : TanhNormalLipschitz := by
  intro Ω L β Λloc Λσ loc σ ν hν hΛl hΛs hl hs m hm m' hm' ω
  refine (W1_map_le ν (measurable_tanhMove _ _) (measurable_tanhMove _ _)).trans ?_
  have hmeas : Measurable fun ξ : Fin 2 → ℝ => ENNReal.ofReal (|ξ 0| + |ξ 1|) :=
    ENNReal.measurable_ofReal.comp
      (((measurable_pi_apply 0).abs).add ((measurable_pi_apply 1).abs))
  have hd0 : 0 ≤ msgDist m m' := Real.sqrt_nonneg _
  calc ∫⁻ ξ, edist (tanhMove (loc m ω) (σ m ω) ξ) (tanhMove (loc m' ω) (σ m' ω) ξ) ∂ν
      ≤ ∫⁻ ξ, (ENNReal.ofReal (60 * Λloc * msgDist m m')
          + ENNReal.ofReal (60 * Λσ * msgDist m m') * ENNReal.ofReal (|ξ 0| + |ξ 1|)) ∂ν := by
        apply lintegral_mono
        intro ξ
        dsimp only
        rw [edist_dist, ← ENNReal.ofReal_mul (by positivity),
          ← ENNReal.ofReal_add (by positivity) (by positivity)]
        apply ENNReal.ofReal_le_ofReal
        refine (dist_tanhMove_le _ _ _ _ ξ).trans ?_
        have a1 := hl m hm m' hm' ω
        have a2 := hs m hm m' hm' ω
        have hξ : 0 ≤ |ξ 0| + |ξ 1| := by positivity
        have : 60 * msgDist (σ m ω) (σ m' ω) * (|ξ 0| + |ξ 1|)
            ≤ 60 * Λσ * msgDist m m' * (|ξ 0| + |ξ 1|) := by
          apply mul_le_mul_of_nonneg_right _ hξ; linarith
        linarith
    _ = _ := by
        rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_const_mul _ hmeas]

theorem lintegral_abs_gaussian_le_one :
    ∫⁻ x, ENNReal.ofReal |x| ∂(ProbabilityTheory.gaussianReal 0 1) ≤ 1 := by
  set N := ProbabilityTheory.gaussianReal 0 1
  have hint2 : Integrable (fun x : ℝ => x ^ 2) N :=
    (ProbabilityTheory.memLp_id_gaussianReal 2).integrable_sq
  have hsq : ∫ x, x ^ 2 ∂N = 1 := by
    have h := ProbabilityTheory.variance_id_gaussianReal (μ := 0) (v := 1)
    rw [ProbabilityTheory.variance_eq_integral measurable_id.aemeasurable] at h
    simpa [ProbabilityTheory.integral_id_gaussianReal] using h
  have hint : Integrable (fun x : ℝ => (1 + x ^ 2) / 2) N :=
    ((integrable_const 1).add hint2).div_const 2
  calc ∫⁻ x, ENNReal.ofReal |x| ∂N ≤ ∫⁻ x, ENNReal.ofReal ((1 + x ^ 2) / 2) ∂N := by
        apply lintegral_mono; intro x
        apply ENNReal.ofReal_le_ofReal
        nlinarith [sq_abs x, sq_nonneg (|x| - 1)]
    _ = ENNReal.ofReal (∫ x, (1 + x ^ 2) / 2 ∂N) :=
        (ofReal_integral_eq_lintegral_ofReal hint
          (Filter.Eventually.of_forall fun x => by positivity)).symm
    _ = 1 := by
        rw [integral_div, integral_add (integrable_const 1) hint2, hsq]
        norm_num

theorem lintegral_noise_le_two :
    ∫⁻ ξ, ENNReal.ofReal (|ξ 0| + |ξ 1|) ∂stdGauss2 ≤ 2 := by
  have hj : ∀ j : Fin 2, ∫⁻ ξ, ENNReal.ofReal |ξ j| ∂stdGauss2 ≤ 1 := by
    intro j
    have hmap : stdGauss2.map (Function.eval j) = ProbabilityTheory.gaussianReal 0 1 := by
      classical
      unfold stdGauss2
      rw [Measure.pi_map_eval]
      simp
    have hmeas : Measurable fun x : ℝ => ENNReal.ofReal |x| :=
      ENNReal.measurable_ofReal.comp measurable_id.abs
    calc ∫⁻ ξ, ENNReal.ofReal |ξ j| ∂stdGauss2
        = ∫⁻ x, ENNReal.ofReal |x| ∂(stdGauss2.map (Function.eval j)) :=
          (lintegral_map hmeas (measurable_pi_apply j)).symm
      _ ≤ 1 := by rw [hmap]; exact lintegral_abs_gaussian_le_one
  calc ∫⁻ ξ, ENNReal.ofReal (|ξ 0| + |ξ 1|) ∂stdGauss2
      = ∫⁻ ξ, (ENNReal.ofReal |ξ 0| + ENNReal.ofReal |ξ 1|) ∂stdGauss2 := by
        congr 1; funext ξ; exact ENNReal.ofReal_add (abs_nonneg _) (abs_nonneg _)
    _ = ∫⁻ ξ, ENNReal.ofReal |ξ 0| ∂stdGauss2 + ∫⁻ ξ, ENNReal.ofReal |ξ 1| ∂stdGauss2 :=
        lintegral_add_left (ENNReal.measurable_ofReal.comp (measurable_pi_apply 0).abs) _
    _ ≤ 1 + 1 := add_le_add (hj 0) (hj 1)
    _ = 2 := by norm_num

theorem tanhNormalSatisfiesB : TanhNormalSatisfiesB := by
  intro Ω L β Λloc Λσ loc σ hΛl hΛs hl hs
  have hprob : IsProbabilityMeasure stdGauss2 := by unfold stdGauss2; infer_instance
  refine ⟨fun m ω => ?_, fun m hm m' hm' ω => ?_⟩
  · unfold tanhNormalLaw
    exact (Measure.isProbabilityMeasure_map_iff (measurable_tanhMove _ _).aemeasurable).mpr hprob
  · refine (tanhNormalLipschitz β Λloc Λσ loc σ stdGauss2 hprob hΛl hΛs hl hs m hm m' hm' ω).trans ?_
    have hd0 : 0 ≤ msgDist m m' := Real.sqrt_nonneg _
    calc ENNReal.ofReal (60 * Λloc * msgDist m m')
          + ENNReal.ofReal (60 * Λσ * msgDist m m') * ∫⁻ ξ, ENNReal.ofReal (|ξ 0| + |ξ 1|) ∂stdGauss2
        ≤ ENNReal.ofReal (60 * Λloc * msgDist m m') + ENNReal.ofReal (60 * Λσ * msgDist m m') * 2 := by
          gcongr; exact lintegral_noise_le_two
      _ = ENNReal.ofReal (60 * (Λloc + 2 * Λσ) * msgDist m m') := by
          rw [show (2 : ℝ≥0∞) = ENNReal.ofReal 2 by simp, ← ENNReal.ofReal_mul (by positivity),
            ← ENNReal.ofReal_add (by positivity) (by positivity)]
          congr 1; ring

end Madre
