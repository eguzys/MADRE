import MadreProof.Statement.Weighted
import MadreProof.Proofs.Airtime
import MadreProof.Proofs.EdgeSet

/-! Weighted radio row: weak duality, slackness, witness scaling, charge antitone, and the
unit-weight special case. -/

open Finset

namespace Madre

variable {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]

omit [DecidableEq V] in
/-- Weak duality for the weighted LP, with the radio-row slack terms. -/
theorem P1W.weak_duality_slack (P : P1W V C D) (κ : D → ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) (hd : P.DualFeasible μ ν lam ρ)
    (f : C → V → V → ℝ) (τ : C → ℝ) (hf : P.Feasible κ f τ) :
    ∑ k, τ k + ∑ i, lam i * ((P.R - κ i) - ∑ v, ∑ w, P.a i v w * ∑ k, f k v w)
      ≤ P.dualObj κ μ lam ρ := by
  obtain ⟨hcap, hflow, hradio, hτ, hf0⟩ := hf
  obtain ⟨hμ, hl, hρ, hred, hτred⟩ := hd
  set E : V → V → ℝ := fun v w => ∑ i, P.a i v w * lam i with hE
  -- step 1: the dual constraint of τ(k)
  have s1 : ∑ k, τ k ≤ ∑ k, ρ k * τ k - ∑ k, ∑ v, P.σ k v * ν v k * τ k := by
    rw [← sum_sub_distrib]
    refine sum_le_sum fun k _ => ?_
    have h1 := hτred k
    have h2 := (hτ k).1
    have : ∑ v, P.σ k v * ν v k * τ k = (∑ v, P.σ k v * ν v k) * τ k := by
      rw [sum_mul]
    rw [this]
    nlinarith
  -- step 2: flow conservation
  have s2 : ∑ k, ∑ v, P.σ k v * ν v k * τ k
      = ∑ k, ∑ v, ∑ w, f k v w * (ν v k - ν w k) := by
    refine sum_congr rfl fun k _ => ?_
    have : ∀ v, P.σ k v * ν v k * τ k = ν v k * (∑ w, f k v w - ∑ w, f k w v) := by
      intro v; rw [hflow v k]; ring
    simp_rw [this, mul_sub, sum_sub_distrib, mul_sum]
    congr 1
    · exact sum_congr rfl fun v _ => sum_congr rfl fun w _ => by ring
    · rw [sum_comm]
      exact sum_congr rfl fun v _ => sum_congr rfl fun w _ => by ring
  -- step 3: the dual constraint of f_k(v,w)
  have s3 : -∑ k, ∑ v, ∑ w, f k v w * (μ v w + E v w)
      ≤ ∑ k, ∑ v, ∑ w, f k v w * (ν v k - ν w k) := by
    rw [← sum_neg_distrib]
    refine sum_le_sum fun k _ => ?_
    rw [← sum_neg_distrib]
    refine sum_le_sum fun v _ => ?_
    rw [← sum_neg_distrib]
    refine sum_le_sum fun w _ => ?_
    have h1 := hred k v w
    have h2 := hf0 k v w
    rw [hE]
    nlinarith
  -- step 4: capacity rows
  have s4 : ∑ k, ∑ v, ∑ w, f k v w * μ v w ≤ ∑ v, ∑ w, μ v w * P.c v w := by
    have : ∑ k, ∑ v, ∑ w, f k v w * μ v w = ∑ v, ∑ w, μ v w * ∑ k, f k v w := by
      rw [sum_comm]
      refine sum_congr rfl fun v _ => ?_
      rw [sum_comm]
      refine sum_congr rfl fun w _ => ?_
      rw [mul_sum]
      exact sum_congr rfl fun k _ => by ring
    rw [this]
    exact sum_le_sum fun v _ => sum_le_sum fun w _ =>
      mul_le_mul_of_nonneg_left (hcap v w) (hμ v w)
  -- step 5: radio rows
  have s5 : ∑ k, ∑ v, ∑ w, f k v w * E v w
      = ∑ i, lam i * ∑ v, ∑ w, P.a i v w * ∑ k, f k v w := by
    have hL : ∑ k, ∑ v, ∑ w, f k v w * E v w
        = ∑ v, ∑ w, ∑ i, ∑ k, f k v w * (P.a i v w * lam i) := by
      simp_rw [hE, mul_sum]
      rw [sum_comm]
      refine sum_congr rfl fun v _ => ?_
      rw [sum_comm]
      refine sum_congr rfl fun w _ => ?_
      rw [sum_comm]
    have hR : ∑ i, lam i * ∑ v, ∑ w, P.a i v w * ∑ k, f k v w
        = ∑ v, ∑ w, ∑ i, ∑ k, f k v w * (P.a i v w * lam i) := by
      simp_rw [mul_sum]
      rw [sum_comm]
      refine sum_congr rfl fun v _ => ?_
      rw [sum_comm]
      refine sum_congr rfl fun w _ => sum_congr rfl fun i _ => sum_congr rfl fun k _ => by ring
    rw [hL, hR]
  have s6 : ∑ i, lam i * ∑ v, ∑ w, P.a i v w * ∑ k, f k v w = ∑ i, lam i * (P.R - κ i)
      - ∑ i, lam i * ((P.R - κ i) - ∑ v, ∑ w, P.a i v w * ∑ k, f k v w) := by
    rw [← sum_sub_distrib]; exact sum_congr rfl fun i _ => by ring
  -- step 6: demand rows
  have s7 : ∑ k, ρ k * τ k ≤ ∑ k, ρ k * P.demand :=
    sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hτ k).2 (hρ k)
  have split : ∑ k, ∑ v, ∑ w, f k v w * (μ v w + E v w)
      = ∑ k, ∑ v, ∑ w, f k v w * μ v w + ∑ k, ∑ v, ∑ w, f k v w * E v w := by
    simp_rw [mul_add, sum_add_distrib]
  unfold P1W.dualObj
  linarith

omit [DecidableEq V] in
theorem P1W.weak_duality (P : P1W V C D) (κ : D → ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) (hd : P.DualFeasible μ ν lam ρ)
    (f : C → V → V → ℝ) (τ : C → ℝ) (hf : P.Feasible κ f τ) :
    ∑ k, τ k ≤ P.dualObj κ μ lam ρ := by
  have h := P.weak_duality_slack κ μ ν lam ρ hd f τ hf
  have : 0 ≤ ∑ i, lam i * ((P.R - κ i) - ∑ v, ∑ w, P.a i v w * ∑ k, f k v w) :=
    sum_nonneg fun i _ => mul_nonneg (hd.2.1 i) (by linarith [hf.2.2.1 i])
  linarith

omit [DecidableEq V] in
theorem P1W.dualObj_charge (P : P1W V C D) (κ : D → ℝ) (μ : V → V → ℝ) (lam : D → ℝ)
    (ρ : C → ℝ) : P.dualObj κ μ lam ρ = P.dualObj 0 μ lam ρ - ∑ i, lam i * κ i := by
  simp only [P1W.dualObj, Pi.zero_apply, sub_zero, mul_sub, sum_sub_distrib]
  ring

omit [DecidableEq V] in
theorem propositionAW_of_dualValue (P : P1W V C D) (b : ℝ) (nbr : D → Finset D) (τ τ0 : ℝ)
    (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ)
    (hτ : IsGreatest (P.values (charge b nbr)) τ) (hd : P.DualFeasible μ ν lam ρ)
    (hsd : P.dualObj 0 μ lam ρ = τ0) :
    τ ≤ τ0 - 8 * b * ∑ i, lam i * (1 + (nbr i).card) := by
  obtain ⟨f, τv, hf, rfl⟩ := hτ.1
  have h := P.weak_duality (charge b nbr) μ ν lam ρ hd f τv hf
  rw [P.dualObj_charge, hsd] at h
  have : ∑ i, lam i * charge b nbr i = 8 * b * ∑ i, lam i * (1 + (nbr i).card) := by
    rw [mul_sum]
    exact sum_congr rfl fun i _ => by unfold charge; ring
  linarith

omit [DecidableEq V] in
theorem complementarySlacknessW_of_dualValue (P : P1W V C D) (κ : D → ℝ)
    (f : C → V → V → ℝ) (τ : C → ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ)
    (ρ : C → ℝ) (hf : P.Feasible κ f τ) (hd : P.DualFeasible μ ν lam ρ)
    (hsd : P.dualObj κ μ lam ρ = ∑ k, τ k) (i : D)
    (hslack : ∑ v, ∑ w, P.a i v w * ∑ k, f k v w < P.R - κ i) : lam i = 0 := by
  have h := P.weak_duality_slack κ μ ν lam ρ hd f τ hf
  rw [hsd] at h
  have hterm : ∀ j ∈ (Finset.univ : Finset D),
      0 ≤ lam j * ((P.R - κ j) - ∑ v, ∑ w, P.a j v w * ∑ k, f k v w) :=
    fun j _ => mul_nonneg (hd.2.1 j) (by linarith [hf.2.2.1 j])
  have hsum0 : ∑ j, lam j * ((P.R - κ j) - ∑ v, ∑ w, P.a j v w * ∑ k, f k v w) = 0 :=
    le_antisymm (by linarith) (sum_nonneg hterm)
  have hi := (sum_eq_zero_iff_of_nonneg hterm).mp hsum0 i (mem_univ i)
  rcases mul_eq_zero.mp hi with h0 | h0
  · exact h0
  · linarith

theorem witnessScalingW : WitnessScalingW := by
  intro V C D _ _ _ _ P κ K x hR hκ hK0 hKR hx
  obtain ⟨f, τ, ⟨hcap, hflow, hradio, hτ, hf0⟩, rfl⟩ := hx
  set θ : ℝ := 1 - K / P.R with hθ
  have hθ0 : 0 ≤ θ := by rw [hθ, sub_nonneg, div_le_one hR]; exact hKR
  have hθ1 : θ ≤ 1 := by rw [hθ]; have : 0 ≤ K / P.R := div_nonneg hK0 hR.le; linarith
  have hθR : θ * P.R = P.R - K := by rw [hθ]; field_simp
  refine ⟨fun k v w => θ * f k v w, fun k => θ * τ k, ⟨?_, ?_, ?_, ?_, ?_⟩, by rw [mul_sum]⟩
  · intro v w
    rw [← mul_sum]
    have h0 : 0 ≤ ∑ k, f k v w := sum_nonneg fun k _ => hf0 k v w
    nlinarith [hcap v w]
  · intro v k
    rw [← mul_sum, ← mul_sum, ← mul_sub, hflow v k]; ring
  · intro i
    have hl := hradio i
    simp only [Pi.zero_apply, sub_zero] at hl
    have e : ∑ v, ∑ w, P.a i v w * ∑ k, θ * f k v w
        = θ * ∑ v, ∑ w, P.a i v w * ∑ k, f k v w := by
      rw [mul_sum]
      refine sum_congr rfl fun v _ => ?_
      rw [mul_sum]
      refine sum_congr rfl fun w _ => ?_
      rw [← mul_sum]; ring
    rw [e]
    have := mul_le_mul_of_nonneg_left hl hθ0
    linarith [(hκ i).2]
  · intro k
    have := hτ k
    constructor
    · exact mul_nonneg hθ0 this.1
    · nlinarith
  · intro k v w; exact mul_nonneg hθ0 (hf0 k v w)

theorem chargeAntitoneW : ChargeAntitoneW := by
  intro V C D _ _ _ _ P κ κ' τ τ' hκ hτ hτ'
  obtain ⟨f, t, ⟨h1, h2, h3, h4, h5⟩, rfl⟩ := hτ'.1
  exact hτ.2 ⟨f, t, ⟨h1, h2, fun i => (h3 i).trans (by linarith [hκ i]), h4, h5⟩, rfl⟩

omit [Fintype D] in
theorem P1.radio_toW (P : P1 V C D) (i : D) (f : C → V → V → ℝ) :
    ∑ v, ∑ w, P.toW.a i v w * ∑ k, f k v w = ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2 := by
  rw [← sum_pairs (P.edgesAt i) (fun e => ∑ k, f k e.1 e.2)]
  refine sum_congr rfl fun v _ => sum_congr rfl fun w _ => ?_
  by_cases h : (v, w) ∈ P.edgesAt i <;> simp [P1.toW, h]

omit [Fintype V] [Fintype C] [DecidableEq V] in
theorem P1.lamTerm_toW [DecidableEq V] (P : P1 V C D) (lam : D → ℝ) (v w : V) :
    ∑ i, P.toW.a i v w * lam i = ∑ i, if (v, w) ∈ P.edgesAt i then lam i else 0 := by
  refine sum_congr rfl fun i _ => ?_
  by_cases h : (v, w) ∈ P.edgesAt i <;> simp [P1.toW, h]

theorem unitWeightSpecialCase : UnitWeightSpecialCase := by
  intro V C D _ _ _ _ P
  refine ⟨fun κ f τ => ?_, fun μ ν lam ρ => ?_, fun κ μ lam ρ => rfl⟩
  · simp only [P1.Feasible, P1W.Feasible, P.radio_toW]
    rfl
  · simp only [P1.DualFeasible, P1W.DualFeasible, P.lamTerm_toW]
    rfl

end Madre
