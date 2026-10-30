import MadreProof.Proofs.WeightedStrongDuality
import MadreProof.Proofs.EdgeSet

/-! The explicit-edge LP with a weighted radio row: same values as the all-pairs weighted LP
with c = 0 and a = 0 off E, optimal duals transfer, and Proposition 1(a) holds for it. -/

open Finset

namespace Madre

variable {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]

def extFW (P : P1EW V C D) (f : C → V × V → ℝ) : C → V → V → ℝ :=
  fun k v w => if (v, w) ∈ P.E then f k (v, w) else 0

omit [Fintype D] in
theorem P1EW.radio_toW (P : P1EW V C D) (i : D) (f' : C → V → V → ℝ) :
    ∑ v, ∑ w, P.toW.a i v w * ∑ k, f' k v w = ∑ e ∈ P.E, P.a i e * ∑ k, f' k e.1 e.2 := by
  rw [← sum_pairs P.E (fun e => P.a i e * ∑ k, f' k e.1 e.2)]
  refine sum_congr rfl fun v _ => sum_congr rfl fun w _ => ?_
  by_cases h : (v, w) ∈ P.E
  · simp [P1EW.toW, h]
  · simp [P1EW.toW, h]

omit [Fintype D] in
theorem P1EW.feasible_toW (P : P1EW V C D) (κ : D → ℝ) (f : C → V × V → ℝ) (τ : C → ℝ)
    (hf : P.Feasible κ f τ) : P.toW.Feasible κ (extFW P f) τ := by
  obtain ⟨hcap, hflow, hradio, hτ, hf0⟩ := hf
  refine ⟨fun v w => ?_, fun v k => ?_, fun i => ?_, hτ, fun k v w => ?_⟩
  · by_cases h : (v, w) ∈ P.E
    · simp only [extFW, P1EW.toW, h, ite_true]; exact hcap _ h
    · simp [extFW, P1EW.toW, h]
  · simp only [P1EW.toW, extFW]
    rw [sum_out P.E (f k) v, sum_in P.E (f k) v]
    exact hflow v k
  · rw [P.radio_toW i (extFW P f)]
    refine le_trans (le_of_eq ?_) (hradio i)
    refine sum_congr rfl fun e he => ?_
    congr 1
    exact sum_congr rfl fun k _ => by simp [extFW, he]
  · by_cases h : (v, w) ∈ P.E
    · simp only [extFW, h, ite_true]; exact hf0 k _ h
    · simp [extFW, h]

omit [Fintype D] in
theorem P1EW.feasible_of_toW (P : P1EW V C D) (κ : D → ℝ) (f' : C → V → V → ℝ) (τ : C → ℝ)
    (hf : P.toW.Feasible κ f' τ) : P.Feasible κ (fun k e => f' k e.1 e.2) τ := by
  obtain ⟨hcap, hflow, hradio, hτ, hf0⟩ := hf
  have hzero : ∀ k v w, (v, w) ∉ P.E → f' k v w = 0 := by
    intro k v w h
    have hc := hcap v w
    simp only [P1EW.toW, h, ite_false] at hc
    have h0 : ∑ k', f' k' v w = 0 := le_antisymm hc (sum_nonneg fun k' _ => hf0 k' v w)
    exact (sum_eq_zero_iff_of_nonneg (fun k' _ => hf0 k' v w)).mp h0 k (mem_univ k)
  have hext : f' = extFW P (fun k e => f' k e.1 e.2) := by
    funext k v w
    by_cases h : (v, w) ∈ P.E
    · simp [extFW, h]
    · simp [extFW, h, hzero k v w h]
  refine ⟨fun e he => ?_, fun v k => ?_, fun i => ?_, hτ, fun k e _ => hf0 k _ _⟩
  · have hc := hcap e.1 e.2
    simp only [P1EW.toW, Prod.mk.eta, he, ite_true] at hc
    exact hc
  · have h := hflow v k
    rw [hext] at h
    simp only [extFW] at h
    rw [sum_out P.E (fun e => f' k e.1 e.2) v, sum_in P.E (fun e => f' k e.1 e.2) v] at h
    exact h
  · have h := hradio i
    rw [P.radio_toW i f'] at h
    exact h

theorem edgeSetValuesEqW : EdgeSetValuesEqW := by
  intro V C D _ _ _ _ P κ
  ext x
  constructor
  · rintro ⟨f, τ, hf, rfl⟩
    exact ⟨extFW P f, τ, P.feasible_toW κ f τ hf, rfl⟩
  · rintro ⟨f', τ, hf, rfl⟩
    exact ⟨fun k e => f' k e.1 e.2, τ, P.feasible_of_toW κ f' τ hf, rfl⟩

def extMuW (P : P1EW V C D) (μ : V × V → ℝ) (ν : V → C → ℝ) : V → V → ℝ :=
  fun v w => if (v, w) ∈ P.E then μ (v, w) else ∑ k, |ν v k - ν w k|

omit [Fintype V] [Fintype C] in
theorem P1EW.lamTerm (P : P1EW V C D) (lam : D → ℝ) (v w : V) :
    (∑ i, P.toW.a i v w * lam i)
      = if (v, w) ∈ P.E then ∑ i, P.a i (v, w) * lam i else 0 := by
  by_cases h : (v, w) ∈ P.E <;> simp [P1EW.toW, h]

theorem P1EW.dualFeasible_toW (P : P1EW V C D) (μ : V × V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) (hd : P.DualFeasible μ ν lam ρ) :
    P.toW.DualFeasible (extMuW P μ ν) ν lam ρ := by
  obtain ⟨hμ, hl, hρ, hred, hτ⟩ := hd
  refine ⟨fun v w => ?_, hl, hρ, fun k v w => ?_, fun k => hτ k⟩
  · by_cases h : (v, w) ∈ P.E
    · simp only [extMuW, h, ite_true]; exact hμ _ h
    · simp only [extMuW, h, ite_false]; exact sum_nonneg fun _ _ => abs_nonneg _
  · rw [P.lamTerm lam v w]
    by_cases h : (v, w) ∈ P.E
    · simp only [extMuW, h, ite_true]
      exact hred k _ h
    · simp only [extMuW, h, ite_false, add_zero]
      have h1 : |ν v k - ν w k| ≤ ∑ k', |ν v k' - ν w k'| :=
        single_le_sum (f := fun k' => |ν v k' - ν w k'|) (fun _ _ => abs_nonneg _) (mem_univ k)
      have h2 := neg_abs_le (ν v k - ν w k)
      linarith

theorem P1EW.dualObj_toW (P : P1EW V C D) (κ : D → ℝ) (μ : V × V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) :
    P.toW.dualObj κ (extMuW P μ ν) lam ρ = P.dualObj κ μ lam ρ := by
  simp only [P1W.dualObj, P1EW.dualObj, P1EW.toW]
  congr 2
  rw [← sum_pairs P.E (fun e => μ e * P.c e)]
  refine sum_congr rfl fun v _ => sum_congr rfl fun w _ => ?_
  by_cases h : (v, w) ∈ P.E <;> simp [extMuW, h]

theorem P1EW.dualFeasible_of_toW (P : P1EW V C D) (μ' : V → V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) (hd : P.toW.DualFeasible μ' ν lam ρ) :
    P.DualFeasible (fun e => μ' e.1 e.2) ν lam ρ ∧
      P.dualObj 0 (fun e => μ' e.1 e.2) lam ρ = P.toW.dualObj 0 μ' lam ρ := by
  obtain ⟨hμ, hl, hρ, hred, hτ⟩ := hd
  refine ⟨⟨fun e _ => hμ _ _, hl, hρ, fun k e he => ?_, fun k => hτ k⟩, ?_⟩
  · have h := hred k e.1 e.2
    rw [P.lamTerm lam e.1 e.2] at h
    simpa [he] using h
  · simp only [P1W.dualObj, P1EW.dualObj, P1EW.toW]
    congr 2
    rw [← sum_pairs P.E (fun e => μ' e.1 e.2 * P.c e)]
    refine sum_congr rfl fun v _ => sum_congr rfl fun w _ => ?_
    by_cases h : (v, w) ∈ P.E <;> simp [h]

theorem propositionAEdgeSetW : PropositionAEdgeSetW := by
  intro V C D _ _ _ _ P b nbr τ τ0 μ ν lam ρ hτ hτ0 hopt
  rw [edgeSetValuesEqW] at hτ hτ0
  have hopt' : P.toW.DualOptimal 0 (extMuW P μ ν) ν lam ρ := by
    refine ⟨P.dualFeasible_toW μ ν lam ρ hopt.1, fun μ' ν' lam' ρ' hd' => ?_⟩
    obtain ⟨hdE, hobj⟩ := P.dualFeasible_of_toW μ' ν' lam' ρ' hd'
    rw [P.dualObj_toW, ← hobj]
    exact hopt.2 _ _ _ _ hdE
  exact propositionAW P.toW b nbr τ τ0 (extMuW P μ ν) ν lam ρ hτ hτ0 hopt'

end Madre
