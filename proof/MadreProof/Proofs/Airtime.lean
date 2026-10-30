import MadreProof.Statement

/-! Proof of Proposition 1(a): weak LP duality for (P1) with the charge on the
right-hand sides of the radio rows. -/

open Finset

namespace Madre

variable {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]

/-- Weak duality for (P1) with the radio-row slack terms kept: the objective plus
∑_i λ(i) (slack of radio row i) is at most the dual objective. -/
theorem P1.weak_duality_slack (P : P1 V C D) (κ : D → ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) (hd : P.DualFeasible μ ν lam ρ)
    (f : C → V → V → ℝ) (τ : C → ℝ) (hf : P.Feasible κ f τ) :
    ∑ k, τ k + ∑ i, lam i * ((P.R - κ i) - ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2)
      ≤ P.dualObj κ μ lam ρ := by
  obtain ⟨hcap, hflow, hradio, hτ, hf0⟩ := hf
  obtain ⟨hμ, hl, hρ, hred, hτred⟩ := hd
  set E : V → V → ℝ := fun v w => ∑ i, if (v, w) ∈ P.edgesAt i then lam i else 0 with hE
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
      = ∑ i, lam i * ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2 := by
    have : ∀ i, ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2
        = ∑ v, ∑ w, if (v, w) ∈ P.edgesAt i then ∑ k, f k v w else 0 := by
      intro i
      rw [← Fintype.sum_prod_type' (f := fun v w => if (v, w) ∈ P.edgesAt i then
        ∑ k, f k v w else 0)]
      rw [sum_ite_mem, univ_inter]
    set g : C → V → V → D → ℝ := fun k v w i =>
      if (v, w) ∈ P.edgesAt i then lam i * f k v w else 0 with hg
    have hL : ∑ k, ∑ v, ∑ w, f k v w * E v w = ∑ v, ∑ w, ∑ i, ∑ k, g k v w i := by
      simp_rw [hE, mul_sum]
      rw [sum_comm]
      refine sum_congr rfl fun v _ => ?_
      rw [sum_comm]
      refine sum_congr rfl fun w _ => ?_
      rw [sum_comm]
      refine sum_congr rfl fun i _ => sum_congr rfl fun k _ => ?_
      rw [hg]
      by_cases h : (v, w) ∈ P.edgesAt i <;> simp [h, mul_comm]
    have hR : ∑ i, lam i * ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2
        = ∑ v, ∑ w, ∑ i, ∑ k, g k v w i := by
      simp_rw [this, mul_sum]
      rw [sum_comm]
      refine sum_congr rfl fun v _ => ?_
      rw [sum_comm]
      refine sum_congr rfl fun w _ => sum_congr rfl fun i _ => ?_
      rw [hg]
      by_cases h : (v, w) ∈ P.edgesAt i <;> simp [h, mul_sum]
    rw [hL, hR]
  have s6 : ∑ i, lam i * ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2 = ∑ i, lam i * (P.R - κ i)
      - ∑ i, lam i * ((P.R - κ i) - ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2) := by
    rw [← sum_sub_distrib]; exact sum_congr rfl fun i _ => by ring
  -- step 6: demand rows
  have s7 : ∑ k, ρ k * τ k ≤ ∑ k, ρ k * P.demand :=
    sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hτ k).2 (hρ k)
  have split : ∑ k, ∑ v, ∑ w, f k v w * (μ v w + E v w)
      = ∑ k, ∑ v, ∑ w, f k v w * μ v w + ∑ k, ∑ v, ∑ w, f k v w * E v w := by
    simp_rw [mul_add, sum_add_distrib]
  unfold P1.dualObj
  linarith

/-- Weak duality for (P1): every feasible objective value is at most the dual objective
of every dual-feasible point, at the same charge `κ`. -/
theorem P1.weak_duality (P : P1 V C D) (κ : D → ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) (hd : P.DualFeasible μ ν lam ρ)
    (f : C → V → V → ℝ) (τ : C → ℝ) (hf : P.Feasible κ f τ) :
    ∑ k, τ k ≤ P.dualObj κ μ lam ρ := by
  have h := P.weak_duality_slack κ μ ν lam ρ hd f τ hf
  have : 0 ≤ ∑ i, lam i * ((P.R - κ i) - ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2) :=
    sum_nonneg fun i _ => mul_nonneg (hd.2.1 i) (by linarith [hf.2.2.1 i])
  linarith

omit [DecidableEq V] in
/-- The dual objective with the charge is the dual objective without it minus ∑ λ κ. -/
theorem P1.dualObj_charge (P : P1 V C D) (κ : D → ℝ) (μ : V → V → ℝ) (lam : D → ℝ)
    (ρ : C → ℝ) : P.dualObj κ μ lam ρ = P.dualObj 0 μ lam ρ - ∑ i, lam i * κ i := by
  simp only [P1.dualObj, Pi.zero_apply, sub_zero, mul_sub, sum_sub_distrib]
  ring

/-- Proposition 1(a) given a dual-feasible point whose dual objective at κ = 0 is τ⁰
(the form used before strong duality was proved). -/
theorem propositionA_of_dualValue (P : P1 V C D) (b : ℝ) (nbr : D → Finset D) (τ τ0 : ℝ)
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

/-- Complementary slackness given a dual-feasible point whose dual objective equals the
primal value. -/
theorem complementarySlackness_of_dualValue (P : P1 V C D) (κ : D → ℝ)
    (f : C → V → V → ℝ) (τ : C → ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ)
    (ρ : C → ℝ) (hf : P.Feasible κ f τ) (hd : P.DualFeasible μ ν lam ρ)
    (hsd : P.dualObj κ μ lam ρ = ∑ k, τ k) (i : D)
    (hslack : ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2 < P.R - κ i) : lam i = 0 := by
  have h := P.weak_duality_slack κ μ ν lam ρ hd f τ hf
  rw [hsd] at h
  have hterm : ∀ j ∈ (Finset.univ : Finset D),
      0 ≤ lam j * ((P.R - κ j) - ∑ e ∈ P.edgesAt j, ∑ k, f k e.1 e.2) :=
    fun j _ => mul_nonneg (hd.2.1 j) (by linarith [hf.2.2.1 j])
  have hsum0 : ∑ j, lam j * ((P.R - κ j) - ∑ e ∈ P.edgesAt j, ∑ k, f k e.1 e.2) = 0 :=
    le_antisymm (by linarith) (sum_nonneg hterm)
  have hi := (sum_eq_zero_iff_of_nonneg hterm).mp hsum0 i (mem_univ i)
  rcases mul_eq_zero.mp hi with h0 | h0
  · exact h0
  · linarith

end Madre
