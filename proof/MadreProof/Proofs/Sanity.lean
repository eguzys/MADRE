import MadreProof.Proofs.Airtime

/-! Sanity checks on the definitions of `MadreProof/Statement.lean`. They are not used
by the proofs. They show that `W1` is not degenerate and that the hypotheses of
Proposition 1(b) can be met (so the proposition is not vacuous). -/

open MeasureTheory Set
open scoped ENNReal

namespace Madre

variable {X : Type} [MetricSpace X] [MeasurableSpace X]

/-- W₁(P, P) = 0: the diagonal coupling costs nothing. -/
theorem W1_self (P : Measure X) : W1 P P = 0 := by
  have hd : Measurable (fun x : X => (x, x)) := measurable_id.prodMk measurable_id
  have h1 : (P.map fun x => (x, x)).map Prod.fst = P := by
    rw [Measure.map_map measurable_fst hd]; exact Measure.map_id
  have h2 : (P.map fun x => (x, x)).map Prod.snd = P := by
    rw [Measure.map_map measurable_snd hd]; exact Measure.map_id
  refine le_antisymm ?_ zero_le
  refine (iInf₂_le (P.map fun x => (x, x)) ⟨h1, h2⟩).trans ?_
  refine (lintegral_map_le _ hd.aemeasurable).trans ?_
  simp

/-- A receiver that ignores the message satisfies the Lipschitz hypothesis of
Proposition 1(b) for every Λ ≥ 0. -/
theorem lipschitz_of_const {Ω : Type} {L : ℕ} (κ : Ω → Measure X) (Λ : ℝ) :
    ∀ m m' : Fin L → ℝ, ∀ ω,
      W1 ((fun (_ : Fin L → ℝ) ω => κ ω) m ω) ((fun (_ : Fin L → ℝ) ω => κ ω) m' ω)
        ≤ ENNReal.ofReal (Λ * msgDist m m') := by
  intro m m' ω
  simp [W1_self]

end Madre

/-! ### Non-vacuity of Proposition 1(a): a tiny LP where every hypothesis holds,
with a positive radio multiplier, and the bound is attained.
Two nodes 0 → 1, one commodity, one drone whose radio row carries the edge (0,1),
c(0,1) = 5, R = 1, D = 2, b = 1/16 and no neighbours, so κ = 1/2. -/

namespace Madre

/-- The tiny instance of (P1). -/
def tinyP : P1 (Fin 2) Unit Unit where
  c := fun v w => if v = 0 ∧ w = 1 then 5 else 0
  σ := fun _ v => if v = 0 then 1 else -1
  R := 1
  demand := 2
  edgesAt := fun _ => {((0 : Fin 2), (1 : Fin 2))}

/-- Flow `x` on the edge (0, 1). -/
def tinyF (x : ℝ) : Unit → Fin 2 → Fin 2 → ℝ := fun _ v w => if v = 0 ∧ w = 1 then x else 0

theorem tiny_feasible (κ : Unit → ℝ) (x : ℝ) (hx0 : 0 ≤ x) (hx : x ≤ 1 - κ ()) (hx2 : x ≤ 2) :
    tinyP.Feasible κ (tinyF x) (fun _ => x) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro v w; fin_cases v <;> fin_cases w <;> (simp [tinyP, tinyF]; try linarith)
  · intro v k; fin_cases v <;> simp [tinyP, tinyF]
  · intro _; simpa [tinyP, tinyF] using hx
  · intro k; exact ⟨hx0, hx2⟩
  · intro k v w; fin_cases v <;> fin_cases w <;> simp [tinyF, hx0]

theorem tiny_dualFeasible :
    tinyP.DualFeasible (fun _ _ => 0) (fun v _ => if v = 0 then -1 else 0) (fun _ => 1)
      (fun _ => 0) := by
  refine ⟨fun _ _ => le_rfl, fun _ => zero_le_one, fun _ => le_rfl, ?_, ?_⟩
  · intro k v w; fin_cases v <;> fin_cases w <;> simp [tinyP]
  · intro k; simp [tinyP]

theorem tiny_dualObj (κ : Unit → ℝ) :
    tinyP.dualObj κ (fun _ _ => 0) (fun _ => 1) (fun _ => 0) = 1 - κ () := by
  simp [P1.dualObj, tinyP]

theorem tiny_isGreatest (κ : Unit → ℝ) (hκ0 : 0 ≤ κ ()) (h0 : 0 ≤ 1 - κ ()) :
    IsGreatest (tinyP.values κ) (1 - κ ()) := by
  refine ⟨⟨tinyF (1 - κ ()), fun _ => 1 - κ (), tiny_feasible κ _ h0 le_rfl (by linarith),
    by simp⟩, ?_⟩
  · rintro y ⟨f, τ, hf, rfl⟩
    have := tinyP.weak_duality κ _ _ _ _ tiny_dualFeasible f τ hf
    rwa [tiny_dualObj] at this

/-- **Every hypothesis of Proposition 1(a) holds at once** (including dual optimality) in the tiny instance, with
λ = 1 > 0, b = 1/16, κ = 1/2, τ = 1/2, τ⁰ = 1, and the bound 1/2 ≤ 1 − 8·(1/16)·1·1 is
attained. -/
theorem propositionA_nonvacuous :
    IsGreatest (tinyP.values (charge (1 / 16) fun _ => ∅)) (1 / 2) ∧
    IsGreatest (tinyP.values 0) 1 ∧
    tinyP.DualOptimal 0 (fun _ _ => 0) (fun v _ => if v = 0 then -1 else 0) (fun _ => 1)
      (fun _ => 0) ∧
    tinyP.dualObj 0 (fun _ _ => 0) (fun _ => 1) (fun _ => 0) = 1 ∧
    (0 : ℝ) < (fun _ : Unit => (1 : ℝ)) () ∧
    (1 / 2 : ℝ) = 1 - 8 * (1 / 16) * ∑ _i : Unit, (1 : ℝ) * (1 + ((∅ : Finset Unit).card : ℝ)) := by
  have hc : charge (1 / 16) (fun _ : Unit => (∅ : Finset Unit)) () = 1 / 2 := by
    simp [charge]; norm_num
  have h0 := tiny_isGreatest 0 le_rfl (by simp)
  simp only [Pi.zero_apply, sub_zero] at h0
  refine ⟨?_, h0, ⟨tiny_dualFeasible, fun μ' ν' lam' ρ' hd' => ?_⟩,
    by rw [tiny_dualObj]; simp, by norm_num, by simp; norm_num⟩
  · have := tiny_isGreatest (charge (1 / 16) fun _ => ∅) (by rw [hc]; norm_num)
      (by rw [hc]; norm_num)
    rwa [hc, show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num] at this
  · obtain ⟨f, τ, hf, hτ⟩ := h0.1
    rw [tiny_dualObj]
    have := tinyP.weak_duality 0 μ' ν' lam' ρ' hd' f τ hf
    simp only [Pi.zero_apply, sub_zero]
    linarith

/-! ### Non-vacuity of Proposition 1(b) with a receiver that depends on the message:
the Dirac kernel of a Λ-Lipschitz map satisfies the W₁-Lipschitz hypothesis. -/

theorem W1_dirac_le {Y : Type} [MetricSpace Y] [MeasurableSpace Y] [BorelSpace Y]
    [SecondCountableTopology Y] (a b : Y) :
    W1 (Measure.dirac a) (Measure.dirac b) ≤ edist a b := by
  have h1 : (Measure.dirac (a, b)).map Prod.fst = Measure.dirac a := by
    rw [Measure.map_dirac' measurable_fst]
  have h2 : (Measure.dirac (a, b)).map Prod.snd = Measure.dirac b := by
    rw [Measure.map_dirac' measurable_snd]
  refine (iInf₂_le (Measure.dirac (a, b)) ⟨h1, h2⟩).trans ?_
  rw [lintegral_dirac' _ (measurable_fst.edist measurable_snd)]

theorem dirac_receiver_lipschitz {Y Ω : Type} [MetricSpace Y] [MeasurableSpace Y]
    [BorelSpace Y] [SecondCountableTopology Y] {L : ℕ} (Λ : ℝ)
    (φ : (Fin L → ℝ) → Ω → Y) (hφ : ∀ m m' (ω : Ω), dist (φ m ω) (φ m' ω) ≤ Λ * msgDist m m') :
    ∀ m m' : Fin L → ℝ, ∀ ω,
      W1 (Measure.dirac (φ m ω)) (Measure.dirac (φ m' ω)) ≤ ENNReal.ofReal (Λ * msgDist m m') :=
  fun m m' ω => (W1_dirac_le _ _).trans (by rw [edist_dist]; exact ENNReal.ofReal_le_ofReal (hφ m m' ω))

/-- A concrete message-dependent receiver: L = 1, moves in ℝ, the move is the received
value itself (h(·|m, ω) = δ_{m₀}); it is 1-Lipschitz for W₁ and gives different laws for
different messages. -/
theorem propositionB_receiver_nonvacuous :
    (∀ m m' : Fin 1 → ℝ, ∀ _ω : Unit,
      W1 (Measure.dirac (m 0)) (Measure.dirac (m' 0)) ≤ ENNReal.ofReal (1 * msgDist m m')) ∧
    Measure.dirac ((fun _ : Fin 1 => (0 : ℝ)) 0) ≠ Measure.dirac ((fun _ : Fin 1 => (1 : ℝ)) 0) := by
  refine ⟨dirac_receiver_lipschitz (Ω := Unit) 1 (fun m _ => m 0) ?_, ?_⟩
  · intro m m' _
    simp [msgDist, Real.dist_eq, Real.sqrt_sq_eq_abs]
  · intro h
    have := congrArg (fun μ : Measure ℝ => μ {0}) h
    simp at this

end Madre
