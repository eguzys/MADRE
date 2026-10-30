import MadreProof.Statement.NoFree
import MadreProof.Proofs.Throughput

/-! Proofs of the prop:nofree mechanisms, the witness bound, cor:air and cor:middle. -/

open MeasureTheory Finset
open scoped ENNReal

namespace Madre

theorem chargeAntitone : ChargeAntitone := by
  intro V C D _ _ _ _ P κ κ' τ τ' hκ hτ hτ'
  obtain ⟨f, t, ⟨h1, h2, h3, h4, h5⟩, rfl⟩ := hτ'.1
  exact hτ.2 ⟨f, t, ⟨h1, h2, fun i => (h3 i).trans (by linarith [hκ i]), h4, h5⟩, rfl⟩

theorem chargeAntitoneInL : ChargeAntitoneInL := by
  intro V C D _ _ _ _ P b nbr τ hb hτ L L' hLL'
  refine chargeAntitone P _ _ _ _ (fun i => ?_) (hτ L) (hτ L')
  unfold charge
  have : (0 : ℝ) ≤ 1 + (nbr i).card := by positivity
  have := hb hLL'
  nlinarith

theorem paddingEmbedding : PaddingEmbedding := by
  intro X O Ω _ _ L' L β Λ hβ hΛ hL K hK
  obtain ⟨g', h', hg', hπ', hLip', rfl⟩ := hK
  let trunc : (Fin L → ℝ) → (Fin L' → ℝ) := fun m j => m (Fin.castLE hL j)
  let pad : (Fin L' → ℝ) → (Fin L → ℝ) := fun m j =>
    if hj : (j : ℕ) < L' then m ⟨j, hj⟩ else 0
  have htp : ∀ m, trunc (pad m) = m := by
    intro m; funext j
    simp [trunc, pad, Fin.castLE]
  have htbox : ∀ m ∈ box L β, trunc m ∈ box L' β := fun m hm j => hm _
  have hpbox : ∀ m ∈ box L' β, pad m ∈ box L β := by
    intro m hm j
    simp only [pad]
    split_ifs
    · exact hm _
    · simpa using hβ
  have hdist : ∀ m m' : Fin L → ℝ, msgDist (trunc m) (trunc m') ≤ msgDist m m' := by
    intro m m'
    unfold msgDist
    apply Real.sqrt_le_sqrt
    have e : ∑ j : Fin L', (trunc m j - trunc m' j) ^ 2
        = ∑ i ∈ (univ : Finset (Fin L')).map (Fin.castLEEmb hL), (m i - m' i) ^ 2 := by
      rw [sum_map]; rfl
    rw [e]
    exact sum_le_sum_of_subset_of_nonneg (subset_univ _) (fun i _ _ => sq_nonneg _)
  refine ⟨fun o => pad (g' o), fun m ω => h' (trunc m) ω, fun o => hpbox _ (hg' o),
    fun m ω => hπ' _ ω, ?_, ?_⟩
  · intro m hm m' hm' ω
    refine (hLip' _ (htbox m hm) _ (htbox m' hm') ω).trans (ENNReal.ofReal_le_ofReal ?_)
    exact mul_le_mul_of_nonneg_left (hdist m m') hΛ
  · funext o ω
    simp only [htp]

theorem witnessScaling : WitnessScaling := by
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
    have e : ∑ e ∈ P.edgesAt i, ∑ k, θ * f k e.1 e.2
        = θ * ∑ e ∈ P.edgesAt i, ∑ k, f k e.1 e.2 := by
      rw [mul_sum]; exact sum_congr rfl fun e _ => by rw [mul_sum]
    rw [e]
    have := mul_le_mul_of_nonneg_left hl hθ0
    linarith [(hκ i).2]
  · intro k
    have := hτ k
    constructor
    · exact mul_nonneg hθ0 this.1
    · nlinarith
  · intro k v w; exact mul_nonneg hθ0 (hf0 k v w)

theorem witnessEpisode : WitnessEpisode := by
  intro Ξ _ P T u thr θ τfull η hθ hut hu
  have h1 : θ * ∑ t ∈ range T, P.E (u t) ≤ ∑ t ∈ range T, P.E (thr t) := by
    rw [mul_sum]
    refine sum_le_sum fun t _ => ?_
    rw [← FinProb.E_const_mul]
    exact P.E_mono (hut t)
  have hT0 : 0 ≤ 1 / (T : ℝ) := by positivity
  have h2 := mul_le_mul_of_nonneg_left h1 hT0
  have h3 := mul_le_mul_of_nonneg_left hu hθ
  have e : 1 / (T : ℝ) * (θ * ∑ t ∈ range T, P.E (u t))
      = θ * (1 / (T : ℝ) * ∑ t ∈ range T, P.E (u t)) := by ring
  linarith

theorem FinProb.E_const {Ξ : Type} [Fintype Ξ] (P : FinProb Ξ) (c : ℝ) :
    P.E (fun _ => c) = c := by
  simp only [FinProb.E, ← sum_mul, P.sum_one, one_mul]

theorem corAir : CorAir := by
  intro Ξ _ P T thr τ0 Λsum b τfull Φ hthr hΦ
  have hT0 : 0 ≤ 1 / (T : ℝ) := by positivity
  have e : 1 / (T : ℝ) * ∑ t ∈ range T, P.E (thr t)
      = P.E (fun ξ => 1 / (T : ℝ) * ∑ t ∈ range T, thr t ξ) := by
    rw [FinProb.E_const_mul, FinProb.E_sum]
  rw [e, ← P.E_const (τfull - Φ)]
  refine P.E_mono fun ξ => ?_
  have h1 : ∑ t ∈ range T, thr t ξ
      ≤ ∑ t ∈ range T, τ0 t ξ - 8 * b * ∑ t ∈ range T, Λsum t ξ := by
    rw [mul_sum, ← sum_sub_distrib]
    exact sum_le_sum fun t _ => hthr t ξ
  have h2 := mul_le_mul_of_nonneg_left h1 hT0
  have := hΦ ξ
  have e2 : 1 / (T : ℝ) * (∑ t ∈ range T, τ0 t ξ - 8 * b * ∑ t ∈ range T, Λsum t ξ)
      = 1 / (T : ℝ) * ∑ t ∈ range T, τ0 t ξ
        - 8 * b * (1 / (T : ℝ) * ∑ t ∈ range T, Λsum t ξ) := by ring
  linarith

theorem corMiddle : CorMiddle := by
  intro A tested Lmin L1 Lmax τ1 Umin Umax hrange hL1 hne1 hb1 hU1 hne2 hb2 hU2 hwit hbdd
    L hL hmax
  obtain ⟨x, hx, hτx⟩ := hwit
  have hstar1 : τ1 ≤ sSup (A L1) := hτx.trans (le_csSup hbdd hx)
  have hmin : sSup (A Lmin) ≤ Umin := csSup_le hne1 hb1
  have hmaxL : sSup (A Lmax) ≤ Umax := csSup_le hne2 hb2
  have hge := hmax L1 hL1
  obtain ⟨h1, h2⟩ := hrange L hL
  constructor
  · rcases lt_or_eq_of_le h1 with h | h
    · exact h
    · subst h; linarith
  · rcases lt_or_eq_of_le h2 with h | h
    · exact h
    · subst h; linarith

end Madre
