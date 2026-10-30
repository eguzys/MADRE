import MadreProof.Proofs.Farkas
import MadreProof.Proofs.Airtime

set_option linter.unusedSimpArgs false

/-! LP strong duality for (P1), from Farkas' lemma: if τ⁰ bounds every feasible objective
value of (P1) with charge κ, some dual-feasible point has dual objective at most τ⁰. -/

open Finset

namespace Madre

variable {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]

omit [Fintype V] [Fintype D] in
theorem sum_edges_eq [Fintype V] (s : Finset (V × V)) (G' : C → V → V → ℝ) :
    ∑ e ∈ s, ∑ k, G' k e.1 e.2 = ∑ k, ∑ v, ∑ w, if (v, w) ∈ s then G' k v w else 0 := by
  rw [sum_comm]
  refine sum_congr rfl fun k _ => ?_
  rw [← Fintype.sum_prod_type' (f := fun v w => if (v, w) ∈ s then G' k v w else 0),
    sum_ite_mem, univ_inter]

namespace SD

/-- Rows of the dual system: one per primal variable f_k(v,w), one per τ(k), one objective. -/
abbrev J (V C : Type) := (C × V × V) ⊕ (C ⊕ Unit)

/-- Nonnegative unknowns of the dual system: μ(v,w), ν⁺(v,k), ν⁻(v,k), λ(i), ρ(k), and the
slacks of the f-rows, of the τ-rows and of the objective row. -/
abbrev G (V C D : Type) :=
  (V × V) ⊕ ((V × C) ⊕ ((V × C) ⊕ (D ⊕ (C ⊕ ((C × V × V) ⊕ (C ⊕ Unit))))))

section
variable (a : G V C D → ℝ)
def μ (v w : V) : ℝ := a (.inl (v, w))
def νp (v : V) (k : C) : ℝ := a (.inr (.inl (v, k)))
def νm (v : V) (k : C) : ℝ := a (.inr (.inr (.inl (v, k))))
def lam (i : D) : ℝ := a (.inr (.inr (.inr (.inl i))))
def ρ (k : C) : ℝ := a (.inr (.inr (.inr (.inr (.inl k)))))
def sf (k : C) (v w : V) : ℝ := a (.inr (.inr (.inr (.inr (.inr (.inl (k, v, w)))))))
def sτ (k : C) : ℝ := a (.inr (.inr (.inr (.inr (.inr (.inr (.inl k)))))))
def u : ℝ := a (.inr (.inr (.inr (.inr (.inr (.inr (.inr ())))))))
end

/-- The left-hand sides of the dual system. -/
def Mfun (P : P1 V C D) (κ : D → ℝ) (a : G V C D → ℝ) : J V C → ℝ
  | .inl (k, v, w) => μ a v w + (νp a v k - νm a v k) - (νp a w k - νm a w k)
      + (∑ i, if (v, w) ∈ P.edgesAt i then lam a i else 0) - sf a k v w
  | .inr (.inl k) => ρ a k - ∑ v, P.σ k v * (νp a v k - νm a v k) - sτ a k
  | .inr (.inr _) => ∑ v, ∑ w, μ a v w * P.c v w + ∑ i, lam a i * (P.R - κ i)
      + ∑ k, ρ a k * P.demand + u a

/-- The same, as a linear map. -/
def M (P : P1 V C D) (κ : D → ℝ) : (G V C D → ℝ) →ₗ[ℝ] (J V C → ℝ) where
  toFun := Mfun P κ
  map_add' a b := by
    funext j
    rcases j with ⟨k, v, w⟩ | k | _
    · simp only [Mfun, μ, νp, νm, lam, sf, Pi.add_apply]
      have : ∀ i, (if (v, w) ∈ P.edgesAt i then a (.inr (.inr (.inr (.inl i))))
          + b (.inr (.inr (.inr (.inl i)))) else 0)
          = (if (v, w) ∈ P.edgesAt i then a (.inr (.inr (.inr (.inl i)))) else 0)
            + (if (v, w) ∈ P.edgesAt i then b (.inr (.inr (.inr (.inl i)))) else 0) := by
        intro i; split_ifs <;> ring
      simp only [this, sum_add_distrib]
      ring
    · simp only [Mfun, ρ, νp, νm, sτ, Pi.add_apply]
      have : ∀ v, P.σ k v * (a (.inr (.inl (v, k))) + b (.inr (.inl (v, k)))
          - (a (.inr (.inr (.inl (v, k)))) + b (.inr (.inr (.inl (v, k))))))
          = P.σ k v * (a (.inr (.inl (v, k))) - a (.inr (.inr (.inl (v, k)))))
            + P.σ k v * (b (.inr (.inl (v, k))) - b (.inr (.inr (.inl (v, k))))) := by
        intro v; ring
      simp only [this, sum_add_distrib]
      ring
    · simp only [Mfun, μ, lam, ρ, u, Pi.add_apply, add_mul, sum_add_distrib]
      ring
  map_smul' c a := by
    funext j
    rcases j with ⟨k, v, w⟩ | k | _
    · simp only [Mfun, μ, νp, νm, lam, sf, Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
      have : ∀ i, (if (v, w) ∈ P.edgesAt i then c * a (.inr (.inr (.inr (.inl i)))) else 0)
          = c * (if (v, w) ∈ P.edgesAt i then a (.inr (.inr (.inr (.inl i)))) else 0) := by
        intro i; split_ifs <;> ring
      simp only [this, ← mul_sum]
      ring
    · simp only [Mfun, ρ, νp, νm, sτ, Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
      have : ∀ v, P.σ k v * (c * a (.inr (.inl (v, k))) - c * a (.inr (.inr (.inl (v, k)))))
          = c * (P.σ k v * (a (.inr (.inl (v, k))) - a (.inr (.inr (.inl (v, k)))))) := by
        intro v; ring
      simp only [this, ← mul_sum]
      ring
    · simp only [Mfun, μ, lam, ρ, u, Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
      simp only [mul_assoc, ← mul_sum]
      ring

open Classical in
theorem M_eq_sum (P : P1 V C D) (κ : D → ℝ) (a : G V C D → ℝ) :
    M P κ a = ∑ g, a g • M P κ (Pi.single g 1) := by
  classical
  conv_lhs => rw [← Finset.univ_sum_single a]
  rw [map_sum]
  refine sum_congr rfl fun g _ => ?_
  rw [← map_smul, ← Pi.single_smul, smul_eq_mul, mul_one]

end SD

open SD in
/-- **LP strong duality for (P1), existence part.** If τ⁰ bounds every feasible objective
value of (P1) with charge κ, some dual-feasible point has dual objective at most τ⁰. -/
theorem P1.exists_dual_le (P : P1 V C D) (κ : D → ℝ) (τ0 : ℝ)
    (hub : ∀ x ∈ P.values κ, x ≤ τ0) :
    ∃ (μ' : V → V → ℝ) (ν' : V → C → ℝ) (lam' : D → ℝ) (ρ' : C → ℝ),
      P.DualFeasible μ' ν' lam' ρ' ∧ P.dualObj κ μ' lam' ρ' ≤ τ0 := by
  classical
  set b : J V C → ℝ := fun j => match j with
    | .inl _ => 0
    | .inr (.inl _) => 1
    | .inr (.inr _) => τ0 with hb
  by_cases hmem : b ∈ Farkas.cone (fun g : G V C D => M P κ (Pi.single g 1))
  · obtain ⟨a, ha, hab⟩ := hmem
    rw [← M_eq_sum] at hab
    refine ⟨μ a, fun v k => νp a v k - νm a v k, lam a, ρ a, ⟨fun v w => ha _, fun i => ha _,
      fun k => ha _, fun k v w => ?_, fun k => ?_⟩, ?_⟩
    · have h := congrFun hab (.inl (k, v, w))
      simp only [hb, M, LinearMap.coe_mk, AddHom.coe_mk, Mfun] at h
      have := ha (.inr (.inr (.inr (.inr (.inr (.inl (k, v, w)))))))
      simp only [sf] at h
      beta_reduce
      linarith
    · have h := congrFun hab (.inr (.inl k))
      simp only [hb, M, LinearMap.coe_mk, AddHom.coe_mk, Mfun] at h
      have := ha (.inr (.inr (.inr (.inr (.inr (.inr (.inl k)))))))
      simp only [sτ] at h
      linarith
    · have h := congrFun hab (.inr (.inr ()))
      simp only [hb, M, LinearMap.coe_mk, AddHom.coe_mk, Mfun] at h
      have := ha (.inr (.inr (.inr (.inr (.inr (.inr (.inr ())))))))
      simp only [u] at h
      unfold P1.dualObj
      linarith
  · exfalso
    obtain ⟨f, hf, hfb⟩ := Farkas.farkas (fun g : G V C D => M P κ (Pi.single g 1)) b hmem
    set y : J V C → ℝ := fun j => f (fun j' => if j = j' then 1 else 0) with hy
    have hfx : ∀ x, f x = ∑ j, x j * y j := by
      intro x
      have := LinearMap.pi_apply_eq_sum_univ f.toLinearMap x
      simpa [smul_eq_mul] using this
    set F : C → V → V → ℝ := fun k v w => y (.inl (k, v, w)) with hF
    set T : C → ℝ := fun k => y (.inr (.inl k)) with hT
    set s : ℝ := y (.inr (.inr ())) with hs
    have pair : ∀ g, f (M P κ (Pi.single g 1)) =
        ∑ k, ∑ v, ∑ w, Mfun P κ (Pi.single g 1) (.inl (k, v, w)) * F k v w
        + ∑ k, Mfun P κ (Pi.single g 1) (.inr (.inl k)) * T k
        + Mfun P κ (Pi.single g 1) (.inr (.inr ())) * s := by
      intro g
      rw [hfx]
      simp only [Fintype.sum_sum_type, Fintype.sum_prod_type, Fintype.sum_unique]
      rw [add_assoc]
      rfl
    have hsf : ∀ k v w, F k v w ≤ 0 := by
      intro k v w
      have h := hf (.inr (.inr (.inr (.inr (.inr (.inl (k, v, w)))))))
      rw [pair] at h
      simp [Mfun, μ, νp, νm, lam, ρ, sf, sτ, u, Pi.single_apply, ite_and, Finset.sum_ite_irrel,
        Finset.sum_ite_eq', sub_mul, ite_mul, Finset.sum_sub_distrib,
        Finset.sum_add_distrib, add_mul] at h
      exact h
    have hsτ : ∀ k, T k ≤ 0 := by
      intro k
      have h := hf (.inr (.inr (.inr (.inr (.inr (.inr (.inl k)))))))
      rw [pair] at h
      simp [Mfun, μ, νp, νm, lam, ρ, sf, sτ, u, Pi.single_apply, ite_and, Finset.sum_ite_irrel,
        Finset.sum_ite_eq', sub_mul, ite_mul, Finset.sum_sub_distrib,
        Finset.sum_add_distrib, add_mul] at h
      exact h
    have hu : 0 ≤ s := by
      have h := hf (.inr (.inr (.inr (.inr (.inr (.inr (.inr ())))))))
      rw [pair] at h
      simp [Mfun, μ, νp, νm, lam, ρ, sf, sτ, u, Pi.single_apply, ite_and, Finset.sum_ite_irrel,
        Finset.sum_ite_eq', sub_mul, ite_mul, Finset.sum_sub_distrib,
        Finset.sum_add_distrib, add_mul] at h
      exact h
    have hμ : ∀ v w, 0 ≤ ∑ k, F k v w + P.c v w * s := by
      intro v w
      have h := hf (.inl (v, w))
      rw [pair] at h
      simp [Mfun, μ, νp, νm, lam, ρ, sf, sτ, u, Pi.single_apply, ite_and, Finset.sum_ite_irrel,
        Finset.sum_ite_eq', sub_mul, ite_mul, Finset.sum_sub_distrib,
        Finset.sum_add_distrib, add_mul] at h
      exact h
    have hνp : ∀ v k, 0 ≤ ∑ w, F k v w - ∑ w, F k w v - P.σ k v * T k := by
      intro v k
      have h := hf (.inr (.inl (v, k)))
      rw [pair] at h
      simp [Mfun, μ, νp, νm, lam, ρ, sf, sτ, u, Pi.single_apply, ite_and, Finset.sum_ite_irrel,
        Finset.sum_ite_eq', sub_mul, ite_mul, Finset.sum_sub_distrib,
        Finset.sum_add_distrib, add_mul] at h
      linarith
    have hνm : ∀ v k, 0 ≤ -(∑ w, F k v w - ∑ w, F k w v - P.σ k v * T k) := by
      intro v k
      have h := hf (.inr (.inr (.inl (v, k))))
      rw [pair] at h
      simp [Mfun, μ, νp, νm, lam, ρ, sf, sτ, u, Pi.single_apply, ite_and, Finset.sum_ite_irrel,
        Finset.sum_ite_eq', sub_mul, ite_mul, Finset.sum_sub_distrib,
        Finset.sum_add_distrib, add_mul] at h
      linarith
    have hlam : ∀ i, 0 ≤ ∑ k, ∑ v, ∑ w, (if (v, w) ∈ P.edgesAt i then F k v w else 0)
        + (P.R - κ i) * s := by
      intro i
      have h := hf (.inr (.inr (.inr (.inl i))))
      rw [pair] at h
      have e : ∀ x1 x2 : V, (∑ x : D, if (x1, x2) ∈ P.edgesAt x then
          (if x = i then (1 : ℝ) else 0) else 0) = if (x1, x2) ∈ P.edgesAt i then 1 else 0 := by
        intro x1 x2
        rw [Finset.sum_eq_single i (fun b _ hb => by simp [hb]) (by simp)]
        simp
      simp [Mfun, μ, νp, νm, lam, ρ, sf, sτ, u, Pi.single_apply, ite_and, Finset.sum_ite_irrel,
        Finset.sum_ite_eq', sub_mul, ite_mul, Finset.sum_sub_distrib,
        Finset.sum_add_distrib, add_mul] at h
      simp only [e, ite_mul, one_mul, zero_mul] at h
      have : (P.R - κ i) * s = P.R * s - κ i * s := by ring
      linarith
    have hρ : ∀ k, 0 ≤ T k + P.demand * s := by
      intro k
      have h := hf (.inr (.inr (.inr (.inr (.inl k)))))
      rw [pair] at h
      simp [Mfun, μ, νp, νm, lam, ρ, sf, sτ, u, Pi.single_apply, ite_and, Finset.sum_ite_irrel,
        Finset.sum_ite_eq', sub_mul, ite_mul, Finset.sum_sub_distrib,
        Finset.sum_add_distrib, add_mul] at h
      exact h
    have hbval : ∑ k, T k + τ0 * s < 0 := by
      rw [hfx] at hfb
      simp only [Fintype.sum_sum_type, Fintype.sum_prod_type, Fintype.sum_unique, hb] at hfb
      simpa [hT, hs] using hfb
    rcases eq_or_lt_of_le hu with hs0 | hspos
    · -- s = 0: then every T k = 0, contradicting f b < 0
      have hT0 : ∀ k, T k = 0 := fun k => by
        have := hρ k; rw [← hs0, mul_zero, add_zero] at this; linarith [hsτ k]
      simp only [hT0, sum_const_zero, ← hs0, mul_zero, add_zero, lt_irrefl] at hbval
    · -- s > 0: (−F/s, −T/s) is feasible with value above τ0
      set fp : C → V → V → ℝ := fun k v w => -F k v w / s with hfp
      set τp : C → ℝ := fun k => -T k / s with hτp
      have hfeas : P.Feasible κ fp τp := by
        refine ⟨fun v w => ?_, fun v k => ?_, fun i => ?_, fun k => ⟨?_, ?_⟩, fun k v w => ?_⟩
        · simp only [hfp, ← sum_div, sum_neg_distrib]
          rw [div_le_iff₀ hspos]; linarith [hμ v w]
        · simp only [hfp, hτp, ← sum_div, sum_neg_distrib]
          rw [div_sub_div_same, mul_div_assoc']
          congr 1
          linarith [hνp v k, hνm v k]
        · rw [sum_edges_eq]
          have e : ∀ k v w, (if (v, w) ∈ P.edgesAt i then fp k v w else 0)
              = -(if (v, w) ∈ P.edgesAt i then F k v w else 0) / s := by
            intro k v w; simp only [hfp]; split_ifs <;> simp
          simp only [e, ← sum_div, sum_neg_distrib]
          rw [div_le_iff₀ hspos]
          have := hlam i
          have : (P.R - κ i) * s = P.R * s - κ i * s := by ring
          linarith
        · simp only [hτp]; exact div_nonneg (by linarith [hsτ k]) hspos.le
        · simp only [hτp]; rw [div_le_iff₀ hspos]; linarith [hρ k]
        · simp only [hfp]; exact div_nonneg (by linarith [hsf k v w]) hspos.le
      have hval := hub _ ⟨fp, τp, hfeas, rfl⟩
      simp only [hτp, ← sum_div, sum_neg_distrib] at hval
      rw [div_le_iff₀ hspos] at hval
      linarith

theorem lpOptimumExists : LPOptimumExists := by
  intro V C D _ _ _ _ P κ hc hD hκ
  set S : Set ((C → V → V → ℝ) × (C → ℝ)) := {p | P.Feasible κ p.1 p.2} with hS
  have hclosed : IsClosed S := by
    have e : S = (⋂ v, ⋂ w, {p : (C → V → V → ℝ) × (C → ℝ) | ∑ k, p.1 k v w ≤ P.c v w}) ∩
        (⋂ v, ⋂ k, {p : (C → V → V → ℝ) × (C → ℝ) |
          ∑ w, p.1 k v w - ∑ w, p.1 k w v = P.σ k v * p.2 k}) ∩
        (⋂ i, {p : (C → V → V → ℝ) × (C → ℝ) |
          ∑ e ∈ P.edgesAt i, ∑ k, p.1 k e.1 e.2 ≤ P.R - κ i}) ∩
        (⋂ k, {p : (C → V → V → ℝ) × (C → ℝ) | 0 ≤ p.2 k}) ∩
        (⋂ k, {p : (C → V → V → ℝ) × (C → ℝ) | p.2 k ≤ P.demand}) ∩
        (⋂ k, ⋂ v, ⋂ w, {p : (C → V → V → ℝ) × (C → ℝ) | 0 ≤ p.1 k v w}) := by
      ext p
      simp only [hS, P1.Feasible, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_iInter,
        forall_and]
      tauto
    rw [e]
    refine (((((isClosed_iInter fun v => isClosed_iInter fun w => isClosed_le ?_ ?_).inter
      (isClosed_iInter fun v => isClosed_iInter fun k => isClosed_eq ?_ ?_)).inter
      (isClosed_iInter fun i => isClosed_le ?_ ?_)).inter
      (isClosed_iInter fun k => isClosed_le ?_ ?_)).inter
      (isClosed_iInter fun k => isClosed_le ?_ ?_)).inter
      (isClosed_iInter fun k => isClosed_iInter fun v => isClosed_iInter fun w =>
        isClosed_le ?_ ?_) <;> fun_prop
  set K : Set ((C → V → V → ℝ) × (C → ℝ)) :=
    (Set.pi Set.univ fun _ => Set.pi Set.univ fun v => Set.pi Set.univ fun w =>
      Set.Icc 0 (P.c v w)) ×ˢ Set.pi Set.univ (fun _ => Set.Icc 0 P.demand) with hK
  have hKc : IsCompact K :=
    (isCompact_univ_pi fun _ => isCompact_univ_pi fun _ => isCompact_univ_pi fun _ =>
      isCompact_Icc).prod (isCompact_univ_pi fun _ => isCompact_Icc)
  have hSK : S ⊆ K := by
    rintro ⟨f, τ⟩ ⟨hcap, -, -, hτ, hf0⟩
    refine ⟨fun k _ v _ w _ => ⟨hf0 k v w, ?_⟩, fun k _ => hτ k⟩
    calc f k v w ≤ ∑ k', f k' v w :=
          single_le_sum (fun k' _ => hf0 k' v w) (mem_univ k)
      _ ≤ P.c v w := hcap v w
  have hSc : IsCompact S := hKc.of_isClosed_subset hclosed hSK
  have hne : S.Nonempty := by
    refine ⟨(0, 0), ?_, ?_, fun i => ?_, fun k => ⟨le_rfl, hD⟩, fun k v w => le_rfl⟩
    · intro v w; simpa using hc v w
    · intro v k; simp
    · simpa using hκ i
  obtain ⟨p, hpS, hpmax⟩ := hSc.exists_isMaxOn hne
    (show Continuous fun p : (C → V → V → ℝ) × (C → ℝ) => ∑ k, p.2 k by fun_prop).continuousOn
  refine ⟨∑ k, p.2 k, ⟨p.1, p.2, hpS, rfl⟩, ?_⟩
  rintro x ⟨f, τ, hf, rfl⟩
  exact hpmax (show (f, τ) ∈ S from hf)

theorem lpStrongDuality : LPStrongDuality := by
  intro V C D _ _ _ _ P κ τ0 hτ0
  obtain ⟨μ, ν, lam, ρ, hd, hle⟩ := P.exists_dual_le κ τ0 hτ0.2
  obtain ⟨f, τ, hf, hτeq⟩ := hτ0.1
  have hge : ∀ μ' ν' lam' ρ', P.DualFeasible μ' ν' lam' ρ' → τ0 ≤ P.dualObj κ μ' lam' ρ' :=
    fun μ' ν' lam' ρ' hd' => hτeq ▸ P.weak_duality κ μ' ν' lam' ρ' hd' f τ hf
  exact ⟨μ, ν, lam, ρ, ⟨hd, fun μ' ν' lam' ρ' hd' => hle.trans (hge μ' ν' lam' ρ' hd')⟩,
    le_antisymm hle (hge μ ν lam ρ hd)⟩

/-- The dual objective of every optimal dual solution equals the primal optimal value. -/
theorem P1.dualOptimal_value (P : P1 V C D) (κ : D → ℝ) (τ0 : ℝ)
    (hτ0 : IsGreatest (P.values κ) τ0) {μ : V → V → ℝ} {ν : V → C → ℝ} {lam : D → ℝ}
    {ρ : C → ℝ} (hopt : P.DualOptimal κ μ ν lam ρ) : P.dualObj κ μ lam ρ = τ0 := by
  obtain ⟨μ', ν', lam', ρ', hopt', hval'⟩ := lpStrongDuality P κ τ0 hτ0
  obtain ⟨f, τ, hf, hτeq⟩ := hτ0.1
  refine le_antisymm ((hopt.2 μ' ν' lam' ρ' hopt'.1).trans hval'.le) ?_
  exact hτeq ▸ P.weak_duality κ μ ν lam ρ hopt.1 f τ hf

theorem propositionA : PropositionA := by
  intro V C D _ _ _ _ P b nbr τ τ0 μ ν lam ρ hτ hτ0 hopt
  exact propositionA_of_dualValue P b nbr τ τ0 μ ν lam ρ hτ hopt.1
    (P.dualOptimal_value 0 τ0 hτ0 hopt)

theorem complementarySlackness : ComplementarySlackness := by
  intro V C D _ _ _ _ P κ f τ μ ν lam ρ hf hτ hopt i hslack
  exact complementarySlackness_of_dualValue P κ f τ μ ν lam ρ hf hopt.1
    (P.dualOptimal_value κ _ hτ hopt) i hslack

end Madre
