import MadreProof.Proofs.WeightedAirtime
import MadreProof.Proofs.Sanity

/-! Non-vacuity of the weighted Proposition 1(a): the tiny LP of `Sanity.lean` with radio weight
2 on its edge (delivery probability p = 1/2, as the scorer charges 1/p), where every hypothesis
holds with λ = 1/2 > 0 and the bound is attained. -/

open Finset

namespace Madre

/-- The tiny LP with the transmit edge (0,1) weighted 2 in the radio row. -/
def tinyW : P1W (Fin 2) Unit Unit where
  c := fun v w => if v = 0 ∧ w = 1 then 5 else 0
  σ := fun _ v => if v = 0 then 1 else -1
  R := 1
  demand := 2
  a := fun _ v w => if v = 0 ∧ w = 1 then 2 else 0

theorem tinyW_feasible (κ : Unit → ℝ) (x : ℝ) (hx0 : 0 ≤ x) (hx : 2 * x ≤ 1 - κ ()) (hx2 : x ≤ 2) :
    tinyW.Feasible κ (tinyF x) (fun _ => x) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro v w; fin_cases v <;> fin_cases w <;> (simp [tinyW, tinyF]; try linarith)
  · intro v k; fin_cases v <;> simp [tinyW, tinyF]
  · intro _; simpa [tinyW, tinyF] using hx
  · intro k; exact ⟨hx0, hx2⟩
  · intro k v w; fin_cases v <;> fin_cases w <;> simp [tinyF, hx0]

theorem tinyW_dualFeasible :
    tinyW.DualFeasible (fun _ _ => 0) (fun v _ => if v = 0 then -1 else 0) (fun _ => 1 / 2)
      (fun _ => 0) := by
  refine ⟨fun _ _ => le_rfl, fun _ => by norm_num, fun _ => le_rfl, ?_, ?_⟩
  · intro k v w; fin_cases v <;> fin_cases w <;> norm_num [tinyW]
  · intro k; simp [tinyW]

theorem tinyW_dualObj (κ : Unit → ℝ) :
    tinyW.dualObj κ (fun _ _ => 0) (fun _ => 1 / 2) (fun _ => 0) = (1 - κ ()) / 2 := by
  simp [P1W.dualObj, tinyW]; ring

theorem tinyW_isGreatest (κ : Unit → ℝ) (hκ0 : 0 ≤ κ ()) (h0 : κ () ≤ 1) :
    IsGreatest (tinyW.values κ) ((1 - κ ()) / 2) := by
  refine ⟨⟨tinyF ((1 - κ ()) / 2), fun _ => (1 - κ ()) / 2,
    tinyW_feasible κ _ (by linarith) (by linarith) (by linarith), by simp⟩, ?_⟩
  rintro y ⟨f, τ, hf, rfl⟩
  have := tinyW.weak_duality κ _ _ _ _ tinyW_dualFeasible f τ hf
  rwa [tinyW_dualObj] at this

/-- **Every hypothesis of the weighted Proposition 1(a) holds at once**, with λ = 1/2 > 0,
b = 1/16, κ = 1/2, τ = 1/4, τ⁰ = 1/2, and the bound 1/4 ≤ 1/2 − 8·(1/16)·(1/2)·1 is attained. -/
theorem propositionAW_nonvacuous :
    IsGreatest (tinyW.values (charge (1 / 16) fun _ => ∅)) (1 / 4) ∧
    IsGreatest (tinyW.values 0) (1 / 2) ∧
    tinyW.DualOptimal 0 (fun _ _ => 0) (fun v _ => if v = 0 then -1 else 0) (fun _ => 1 / 2)
      (fun _ => 0) ∧
    (0 : ℝ) < 1 / 2 ∧
    (1 / 4 : ℝ) = 1 / 2 - 8 * (1 / 16) *
      ∑ _i : Unit, (1 / 2 : ℝ) * (1 + ((∅ : Finset Unit).card : ℝ)) := by
  have hc : charge (1 / 16) (fun _ : Unit => (∅ : Finset Unit)) () = 1 / 2 := by
    simp [charge]; norm_num
  have h0 := tinyW_isGreatest 0 le_rfl (by simp)
  simp only [Pi.zero_apply, sub_zero] at h0
  refine ⟨?_, h0, ⟨tinyW_dualFeasible, fun μ' ν' lam' ρ' hd' => ?_⟩, by norm_num,
    by simp; norm_num⟩
  · have := tinyW_isGreatest (charge (1 / 16) fun _ => ∅) (by rw [hc]; norm_num)
      (by rw [hc]; norm_num)
    rwa [hc, show (1 - 1 / 2 : ℝ) / 2 = 1 / 4 by norm_num] at this
  · obtain ⟨f, τ, hf, hτ⟩ := h0.1
    rw [tinyW_dualObj]
    have := tinyW.weak_duality 0 μ' ν' lam' ρ' hd' f τ hf
    simp only [Pi.zero_apply, sub_zero]
    linarith

end Madre
