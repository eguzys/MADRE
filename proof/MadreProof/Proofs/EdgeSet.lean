import MadreProof.Statement.EdgeSet
import MadreProof.Proofs.StrongDuality

/-! Proofs for the explicit-edge LP: it has the same feasible values as the all-pairs LP with
c = 0 off E, optimal duals transfer, and Proposition 1(a) holds for it. -/

open Finset

namespace Madre

section Sums
variable {V : Type} [Fintype V] [DecidableEq V]
theorem sum_filter_eq_univ (E : Finset (V × V)) (p : V × V → Prop) [DecidablePred p]
    (g : V × V → ℝ) :
    ∑ e ∈ E.filter p, g e = ∑ e, if e ∈ E then (if p e then g e else 0) else 0 := by
  rw [Finset.sum_filter, Finset.sum_ite_mem, univ_inter]

theorem sum_out (E : Finset (V × V)) (g : V × V → ℝ) (v : V) :
    ∑ w, (if (v, w) ∈ E then g (v, w) else 0) = ∑ e ∈ E.filter (fun e => e.1 = v), g e := by
  rw [sum_filter_eq_univ, Fintype.sum_prod_type]
  have h : ∀ a b, (if (a, b) ∈ E then (if (a, b).1 = v then g (a, b) else 0) else 0)
      = if a = v then (if (a, b) ∈ E then g (a, b) else 0) else 0 := by
    intro a b; by_cases h1 : (a, b) ∈ E <;> by_cases h2 : a = v <;> simp [h1, h2]
  simp only [h, Finset.sum_ite_irrel, sum_const_zero, Finset.sum_ite_eq', mem_univ, ite_true]

theorem sum_in (E : Finset (V × V)) (g : V × V → ℝ) (v : V) :
    ∑ w, (if (w, v) ∈ E then g (w, v) else 0) = ∑ e ∈ E.filter (fun e => e.2 = v), g e := by
  rw [sum_filter_eq_univ, Fintype.sum_prod_type, Finset.sum_comm]
  have h : ∀ b a, (if (a, b) ∈ E then (if (a, b).2 = v then g (a, b) else 0) else 0)
      = if b = v then (if (a, b) ∈ E then g (a, b) else 0) else 0 := by
    intro b a; by_cases h1 : (a, b) ∈ E <;> by_cases h2 : b = v <;> simp [h1, h2]
  simp only [h, Finset.sum_ite_irrel, sum_const_zero, Finset.sum_ite_eq', mem_univ, ite_true]

theorem sum_pairs (E : Finset (V × V)) (g : V × V → ℝ) :
    ∑ v, ∑ w, (if (v, w) ∈ E then g (v, w) else 0) = ∑ e ∈ E, g e := by
  rw [← Fintype.sum_prod_type' (f := fun v w => if (v, w) ∈ E then g (v, w) else 0),
    Finset.sum_ite_mem, univ_inter]
end Sums

variable {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]

/-- Extension of an edge flow by zero off E. -/
def extF (P : P1E V C D) (f : C → V × V → ℝ) : C → V → V → ℝ :=
  fun k v w => if (v, w) ∈ P.E then f k (v, w) else 0

omit [Fintype D] in
theorem P1E.feasible_toP1 (P : P1E V C D) (κ : D → ℝ) (f : C → V × V → ℝ) (τ : C → ℝ)
    (hf : P.Feasible κ f τ) : P.toP1.Feasible κ (extF P f) τ := by
  obtain ⟨hcap, hflow, hradio, hτ, hf0⟩ := hf
  refine ⟨fun v w => ?_, fun v k => ?_, fun i => ?_, hτ, fun k v w => ?_⟩
  · by_cases h : (v, w) ∈ P.E
    · simp only [extF, P1E.toP1, h, ite_true]; exact hcap _ h
    · simp [extF, P1E.toP1, h]
  · simp only [P1E.toP1, extF]
    rw [sum_out P.E (f k) v, sum_in P.E (f k) v]
    exact hflow v k
  · simp only [P1E.toP1]
    refine le_trans (le_of_eq ?_) (hradio i)
    refine sum_congr rfl fun e he => sum_congr rfl fun k _ => ?_
    simp [extF, (mem_inter.mp he).1]
  · by_cases h : (v, w) ∈ P.E
    · simp only [extF, h, ite_true]; exact hf0 k _ h
    · simp [extF, h]

omit [Fintype D] in
theorem P1E.feasible_of_toP1 (P : P1E V C D) (κ : D → ℝ) (f' : C → V → V → ℝ) (τ : C → ℝ)
    (hf : P.toP1.Feasible κ f' τ) : P.Feasible κ (fun k e => f' k e.1 e.2) τ := by
  obtain ⟨hcap, hflow, hradio, hτ, hf0⟩ := hf
  have hzero : ∀ k v w, (v, w) ∉ P.E → f' k v w = 0 := by
    intro k v w h
    have hc := hcap v w
    simp only [P1E.toP1, h, ite_false] at hc
    have h0 : ∑ k', f' k' v w = 0 := le_antisymm hc (sum_nonneg fun k' _ => hf0 k' v w)
    exact (sum_eq_zero_iff_of_nonneg (fun k' _ => hf0 k' v w)).mp h0 k (mem_univ k)
  have hext : f' = extF P (fun k e => f' k e.1 e.2) := by
    funext k v w
    by_cases h : (v, w) ∈ P.E
    · simp [extF, h]
    · simp [extF, h, hzero k v w h]
  refine ⟨fun e he => ?_, fun v k => ?_, fun i => hradio i, hτ, fun k e _ => hf0 k _ _⟩
  · have hc := hcap e.1 e.2
    simp only [P1E.toP1, Prod.mk.eta, he, ite_true] at hc
    exact hc
  · have h := hflow v k
    rw [hext] at h
    simp only [extF] at h
    rw [sum_out P.E (fun e => f' k e.1 e.2) v, sum_in P.E (fun e => f' k e.1 e.2) v] at h
    exact h

theorem edgeSetValuesEq : EdgeSetValuesEq := by
  intro V C D _ _ _ _ P κ
  ext x
  constructor
  · rintro ⟨f, τ, hf, rfl⟩
    exact ⟨extF P f, τ, P.feasible_toP1 κ f τ hf, rfl⟩
  · rintro ⟨f', τ, hf, rfl⟩
    exact ⟨fun k e => f' k e.1 e.2, τ, P.feasible_of_toP1 κ f' τ hf, rfl⟩

/-- Extension of an edge dual: on a non-edge, μ is large enough at zero cost (c = 0 there). -/
def extMu (P : P1E V C D) (μ : V × V → ℝ) (ν : V → C → ℝ) : V → V → ℝ :=
  fun v w => if (v, w) ∈ P.E then μ (v, w) else ∑ k, |ν v k - ν w k|

omit [Fintype V] [Fintype C] in
theorem P1E.lam_term (P : P1E V C D) (lam : D → ℝ) {v w : V} (h : (v, w) ∈ P.E) :
    (∑ i, if (v, w) ∈ P.toP1.edgesAt i then lam i else 0)
      = ∑ i, if (v, w) ∈ P.edgesAt i then lam i else 0 := by
  refine sum_congr rfl fun i _ => ?_
  simp [P1E.toP1, mem_inter, h]

theorem P1E.dualFeasible_toP1 (P : P1E V C D) (μ : V × V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) (hd : P.DualFeasible μ ν lam ρ) :
    P.toP1.DualFeasible (extMu P μ ν) ν lam ρ := by
  obtain ⟨hμ, hl, hρ, hred, hτ⟩ := hd
  refine ⟨fun v w => ?_, hl, hρ, fun k v w => ?_, fun k => hτ k⟩
  · by_cases h : (v, w) ∈ P.E
    · simp only [extMu, h, ite_true]; exact hμ _ h
    · simp only [extMu, h, ite_false]; exact sum_nonneg fun _ _ => abs_nonneg _
  · by_cases h : (v, w) ∈ P.E
    · rw [P.lam_term lam h]
      simp only [extMu, h, ite_true]
      exact hred k _ h
    · have hl0 : (∑ i, if (v, w) ∈ P.toP1.edgesAt i then lam i else 0) = 0 :=
        sum_eq_zero fun i _ => by simp [P1E.toP1, mem_inter, h]
      rw [hl0]
      simp only [extMu, h, ite_false]
      have h1 : |ν v k - ν w k| ≤ ∑ k', |ν v k' - ν w k'| :=
        single_le_sum (f := fun k' => |ν v k' - ν w k'|) (fun _ _ => abs_nonneg _) (mem_univ k)
      have h2 := neg_abs_le (ν v k - ν w k)
      linarith

theorem P1E.dualObj_toP1 (P : P1E V C D) (κ : D → ℝ) (μ : V × V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) :
    P.toP1.dualObj κ (extMu P μ ν) lam ρ = P.dualObj κ μ lam ρ := by
  simp only [P1.dualObj, P1E.dualObj, P1E.toP1]
  congr 2
  rw [← sum_pairs P.E (fun e => μ e * P.c e)]
  refine sum_congr rfl fun v _ => sum_congr rfl fun w _ => ?_
  by_cases h : (v, w) ∈ P.E <;> simp [extMu, h]

theorem P1E.dualFeasible_of_toP1 (P : P1E V C D) (μ' : V → V → ℝ) (ν : V → C → ℝ)
    (lam : D → ℝ) (ρ : C → ℝ) (hd : P.toP1.DualFeasible μ' ν lam ρ) :
    P.DualFeasible (fun e => μ' e.1 e.2) ν lam ρ ∧
      P.dualObj 0 (fun e => μ' e.1 e.2) lam ρ = P.toP1.dualObj 0 μ' lam ρ := by
  obtain ⟨hμ, hl, hρ, hred, hτ⟩ := hd
  refine ⟨⟨fun e _ => hμ _ _, hl, hρ, fun k e he => ?_, fun k => hτ k⟩, ?_⟩
  · have h := hred k e.1 e.2
    rw [P.lam_term lam (by simpa using he)] at h
    simpa using h
  · simp only [P1.dualObj, P1E.dualObj, P1E.toP1]
    congr 2
    rw [← sum_pairs P.E (fun e => μ' e.1 e.2 * P.c e)]
    refine sum_congr rfl fun v _ => sum_congr rfl fun w _ => ?_
    by_cases h : (v, w) ∈ P.E <;> simp [h]

theorem propositionAEdgeSet : PropositionAEdgeSet := by
  intro V C D _ _ _ _ P b nbr τ τ0 μ ν lam ρ hτ hτ0 hopt
  rw [edgeSetValuesEq] at hτ hτ0
  have hopt' : P.toP1.DualOptimal 0 (extMu P μ ν) ν lam ρ := by
    refine ⟨P.dualFeasible_toP1 μ ν lam ρ hopt.1, fun μ' ν' lam' ρ' hd' => ?_⟩
    obtain ⟨hdE, hobj⟩ := P.dualFeasible_of_toP1 μ' ν' lam' ρ' hd'
    rw [P.dualObj_toP1, ← hobj]
    exact hopt.2 _ _ _ _ hdE
  exact propositionA P.toP1 b nbr τ τ0 (extMu P μ ν) ν lam ρ hτ hτ0 hopt'

end Madre
