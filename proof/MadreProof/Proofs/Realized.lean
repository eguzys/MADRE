import MadreProof.Statement.Realized
import MadreProof.Proofs.Receivers

/-! Proofs for the realized move: the clamp and the wall reflection are 1-Lipschitz. -/

open MeasureTheory Real
open scoped ENNReal

namespace Madre

theorem dist_squashMove_le (sq : ℝ → ℝ) (hsq : ∀ a b, |sq a - sq b| ≤ |a - b|)
    (l s l' s' ξ : Fin 2 → ℝ) :
    dist (diskClip 60 (WithLp.toLp 2 (fun j => 60 * sq (l j + s j * ξ j))))
      (diskClip 60 (WithLp.toLp 2 (fun j => 60 * sq (l' j + s' j * ξ j))))
      ≤ 60 * msgDist l l' + 60 * msgDist s s' * (|ξ 0| + |ξ 1|) := by
  refine (dist_diskClip_le (by norm_num) _ _).trans ?_
  rw [EuclideanSpace.dist_eq, Fin.sum_univ_two]
  simp only [Real.dist_eq, sq_abs]
  set d0 := (l 0 + s 0 * ξ 0) - (l' 0 + s' 0 * ξ 0)
  set d1 := (l 1 + s 1 * ξ 1) - (l' 1 + s' 1 * ξ 1)
  have t0 : (60 * sq (l 0 + s 0 * ξ 0) - 60 * sq (l' 0 + s' 0 * ξ 0)) ^ 2 ≤ (60 * d0) ^ 2 := by
    rw [← sq_abs, ← sq_abs (60 * d0)]
    have := hsq (l 0 + s 0 * ξ 0) (l' 0 + s' 0 * ξ 0)
    have e : |60 * sq (l 0 + s 0 * ξ 0) - 60 * sq (l' 0 + s' 0 * ξ 0)|
        = 60 * |sq (l 0 + s 0 * ξ 0) - sq (l' 0 + s' 0 * ξ 0)| := by
      rw [← mul_sub, abs_mul]; norm_num
    rw [e, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 60)]
    exact pow_le_pow_left₀ (by positivity) (by linarith) 2
  have t1 : (60 * sq (l 1 + s 1 * ξ 1) - 60 * sq (l' 1 + s' 1 * ξ 1)) ^ 2 ≤ (60 * d1) ^ 2 := by
    rw [← sq_abs, ← sq_abs (60 * d1)]
    have := hsq (l 1 + s 1 * ξ 1) (l' 1 + s' 1 * ξ 1)
    have e : |60 * sq (l 1 + s 1 * ξ 1) - 60 * sq (l' 1 + s' 1 * ξ 1)|
        = 60 * |sq (l 1 + s 1 * ξ 1) - sq (l' 1 + s' 1 * ξ 1)| := by
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
  calc Real.sqrt ((60 * sq (l 0 + s 0 * ξ 0) - 60 * sq (l' 0 + s' 0 * ξ 0)) ^ 2
        + (60 * sq (l 1 + s 1 * ξ 1) - 60 * sq (l' 1 + s' 1 * ξ 1)) ^ 2)
      ≤ Real.sqrt ((60 * d0) ^ 2 + (60 * d1) ^ 2) := Real.sqrt_le_sqrt (by linarith)
    _ = 60 * Real.sqrt (d0 ^ 2 + d1 ^ 2) := hsq
    _ ≤ 60 * (msgDist l l' + msgDist s s' * (|ξ 0| + |ξ 1|)) := by gcongr
    _ = _ := by ring

theorem abs_reflect1_sub_le {lo hi : ℝ} (h : lo ≤ hi) (x y : ℝ) :
    |reflect1 lo hi x - reflect1 lo hi y| ≤ |x - y| := by
  have a1 := le_abs_self (x - y)
  have a2 := neg_abs_le (x - y)
  unfold reflect1
  split_ifs <;> rw [abs_le] <;> constructor <;> linarith

theorem abs_clamp_sub_le (c u v : ℝ) :
    |max (-c) (min c u) - max (-c) (min c v)| ≤ |u - v| := by
  have a1 := le_abs_self (u - v)
  have a2 := neg_abs_le (u - v)
  simp only [max_def, min_def]
  split_ifs <;> rw [abs_le] <;> constructor <;> linarith

theorem abs_safeTanh_sub_le (c a b : ℝ) : |safeTanh c a - safeTanh c b| ≤ |a - b| :=
  (abs_clamp_sub_le c _ _).trans (abs_tanh_sub_le a b)

theorem dist_realizedMove_le {c lo hi : ℝ} (h : lo ≤ hi) (p l s l' s' ξ : Fin 2 → ℝ) :
    dist (realizedMove c lo hi p l s ξ) (realizedMove c lo hi p l' s' ξ)
      ≤ 60 * msgDist l l' + 60 * msgDist s s' * (|ξ 0| + |ξ 1|) := by
  refine le_trans ?_ (dist_squashMove_le (safeTanh c) (abs_safeTanh_sub_le c) l s l' s' ξ)
  set u := commandedMove c l s ξ
  set u' := commandedMove c l' s' ξ
  change dist (realizedMove c lo hi p l s ξ) (realizedMove c lo hi p l' s' ξ) ≤ dist u u'
  rw [EuclideanSpace.dist_eq, EuclideanSpace.dist_eq]
  apply Real.sqrt_le_sqrt
  refine Finset.sum_le_sum fun j _ => ?_
  simp only [realizedMove, Real.dist_eq, sq_abs]
  have hr := abs_reflect1_sub_le h (p j + u j) (p j + u' j)
  have e : reflect1 lo hi (p j + u j) - p j - (reflect1 lo hi (p j + u' j) - p j)
      = reflect1 lo hi (p j + u j) - reflect1 lo hi (p j + u' j) := by ring
  have e2 : p j + u j - (p j + u' j) = u j - u' j := by ring
  rw [e2] at hr
  rw [e, ← sq_abs, ← sq_abs (u j - u' j)]
  exact pow_le_pow_left₀ (abs_nonneg _) hr 2

theorem measurable_reflect1 (lo hi : ℝ) : Measurable (reflect1 lo hi) := by
  unfold reflect1
  exact Measurable.ite (measurableSet_lt measurable_id measurable_const)
    (measurable_const.sub measurable_id)
    (Measurable.ite (measurableSet_lt measurable_const measurable_id)
      (measurable_const.sub measurable_id) measurable_id)

theorem measurable_commandedMove (c : ℝ) (l s : Fin 2 → ℝ) : Measurable (commandedMove c l s) := by
  unfold commandedMove
  refine (measurable_diskClip 60).comp ((WithLp.measurable_toLp 2 _).comp ?_)
  refine measurable_pi_iff.2 fun j => ?_
  have hst : Measurable (safeTanh c) := by
    unfold safeTanh
    exact measurable_const.max (measurable_const.min continuous_tanh'.measurable)
  exact measurable_const.mul (hst.comp
    (measurable_const.add (measurable_const.mul (measurable_pi_apply j))))

theorem measurable_realizedMove (c lo hi : ℝ) (p l s : Fin 2 → ℝ) :
    Measurable (realizedMove c lo hi p l s) := by
  unfold realizedMove
  refine (WithLp.measurable_toLp 2 _).comp (measurable_pi_iff.2 fun j => ?_)
  have hcoord : Measurable fun x : Plane => x j :=
    (EuclideanSpace.proj (𝕜 := ℝ) (ι := Fin 2) j).continuous.measurable
  exact ((measurable_reflect1 lo hi).comp
    (measurable_const.add (hcoord.comp (measurable_commandedMove c l s)))).sub measurable_const

theorem realizedSatisfiesB : RealizedSatisfiesB := by
  intro Ω L β Λloc Λσ c lo hi loc σ pos hlh hΛl hΛs hl hs
  have hprob : IsProbabilityMeasure stdGauss2 := by unfold stdGauss2; infer_instance
  refine ⟨fun m ω => ?_, fun m hm m' hm' ω => ?_⟩
  · unfold realizedLaw
    exact (Measure.isProbabilityMeasure_map_iff
      (measurable_realizedMove _ _ _ _ _ _).aemeasurable).mpr hprob
  · refine (W1_map_le stdGauss2 (measurable_realizedMove _ _ _ _ _ _)
      (measurable_realizedMove _ _ _ _ _ _)).trans ?_
    have hmeas : Measurable fun ξ : Fin 2 → ℝ => ENNReal.ofReal (|ξ 0| + |ξ 1|) :=
      ENNReal.measurable_ofReal.comp
        (((measurable_pi_apply 0).abs).add ((measurable_pi_apply 1).abs))
    have hd0 : 0 ≤ msgDist m m' := Real.sqrt_nonneg _
    calc ∫⁻ ξ, edist (realizedMove c lo hi (pos ω) (loc m ω) (σ m ω) ξ)
            (realizedMove c lo hi (pos ω) (loc m' ω) (σ m' ω) ξ) ∂stdGauss2
        ≤ ∫⁻ ξ, (ENNReal.ofReal (60 * Λloc * msgDist m m')
            + ENNReal.ofReal (60 * Λσ * msgDist m m') * ENNReal.ofReal (|ξ 0| + |ξ 1|))
            ∂stdGauss2 := by
          apply lintegral_mono
          intro ξ
          dsimp only
          rw [edist_dist, ← ENNReal.ofReal_mul (by positivity),
            ← ENNReal.ofReal_add (by positivity) (by positivity)]
          apply ENNReal.ofReal_le_ofReal
          refine (dist_realizedMove_le hlh _ _ _ _ _ ξ).trans ?_
          have a1 := hl m hm m' hm' ω
          have a2 := hs m hm m' hm' ω
          have hξ : 0 ≤ |ξ 0| + |ξ 1| := by positivity
          have : 60 * msgDist (σ m ω) (σ m' ω) * (|ξ 0| + |ξ 1|)
              ≤ 60 * Λσ * msgDist m m' * (|ξ 0| + |ξ 1|) := by
            apply mul_le_mul_of_nonneg_right _ hξ; linarith
          linarith
      _ = ENNReal.ofReal (60 * Λloc * msgDist m m')
            + ENNReal.ofReal (60 * Λσ * msgDist m m') *
              ∫⁻ ξ, ENNReal.ofReal (|ξ 0| + |ξ 1|) ∂stdGauss2 := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
            lintegral_const_mul _ hmeas]
      _ ≤ ENNReal.ofReal (60 * Λloc * msgDist m m') + ENNReal.ofReal (60 * Λσ * msgDist m m') * 2 := by
          gcongr; exact lintegral_noise_le_two
      _ = ENNReal.ofReal (60 * (Λloc + 2 * Λσ) * msgDist m m') := by
          rw [show (2 : ℝ≥0∞) = ENNReal.ofReal 2 by simp, ← ENNReal.ofReal_mul (by positivity),
            ← ENNReal.ofReal_add (by positivity) (by positivity)]
          congr 1; ring

theorem realizedDeterministicLipschitz : RealizedDeterministicLipschitz := by
  intro Ω L β Λloc c lo hi loc σ pos hlh hl m hm m' hm' ω
  refine (dist_realizedMove_le hlh _ _ _ _ _ 0).trans ?_
  have := hl m hm m' hm' ω
  simp only [Pi.zero_apply, abs_zero, add_zero, mul_zero]
  linarith

end Madre
