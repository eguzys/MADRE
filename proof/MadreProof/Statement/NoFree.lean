/-
# Theory note, Proposition prop:nofree, the witness bound, Corollaries cor:air and
# cor:middle: formal statements

Statements only, no proofs (they are in `MadreProof/Proofs/NoFree.lean`).

The note's prop:nofree ("the model contains scenarios in which τ*(L) is largest at the
smallest tested size and scenarios in which it is largest at the largest one") quantifies
over all scenarios of an informally specified model, and τ*(L) is a supremum over
encoder/receiver pairs of an episode throughput that the note never constructs from the
pair. It is formalized here as its two mechanisms, each exactly:
* `ChargeAntitone`: with the charge on the radio rows, the optimal value of (P1) can only
  stay equal or drop as b(L) grows (the "smallest size is best" scenario).
* `PaddingEmbedding`: every encoder/receiver pair at size L' ≤ L is realized at size L with
  the same laws of moves, the same range [-β,β] and the same Lipschitz constant (the
  "largest size is best" scenario, when the charge never binds and messages always arrive).
cor:middle is formalized as the logical step of its proof (`CorMiddle`), with its three
ingredients proved separately: Theorem thm:floor (iii) (`TheoremFloorIII`), cor:air
(`CorAir`) and the witness bound (`WitnessScaling`, `WitnessEpisode`).
-/
import MadreProof.Statement.Throughput

open MeasureTheory
open scoped ENNReal

namespace Madre

/-- **prop:nofree, small side.** If the charge grows pointwise (κ ≤ κ'), the optimal value
of (P1) does not increase. With κ = κ(i) = 8 b(L)(1 + |𝒟_t(i)|) and b nondecreasing in L,
the optimal value is nonincreasing in L at fixed positions (`ChargeAntitoneInL`). -/
def ChargeAntitone : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1 V C D) (κ κ' : D → ℝ) (τ τ' : ℝ),
    (∀ i, κ i ≤ κ' i) → IsGreatest (P.values κ) τ → IsGreatest (P.values κ') τ' → τ' ≤ τ

/-- The same with κ = charge b(L): if b(L) is nondecreasing in L and τ(L) is the optimal
value of (P1) at charge 8 b(L)(1 + |𝒟_t(i)|), then τ(L) is nonincreasing in L. -/
def ChargeAntitoneInL : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1 V C D) (b : ℕ → ℝ) (nbr : D → Finset D) (τ : ℕ → ℝ),
    Monotone b → (∀ L, IsGreatest (P.values (charge (b L) nbr)) (τ L)) → Antitone τ

/-- The family of laws of moves o, ω ↦ h(·|g(o), ω) that some encoder g into [-β,β]^L and
some receiver h (probability measures, W₁-Λ-Lipschitz in m on [-β,β]^L) produce. -/
def AdmissibleLaws (X O Ω : Type) [MetricSpace X] [MeasurableSpace X]
    (L : ℕ) (β Λ : ℝ) : Set (O → Ω → Measure X) :=
  {K | ∃ (g : O → Fin L → ℝ) (h : (Fin L → ℝ) → Ω → Measure X),
    (∀ o, g o ∈ box L β) ∧ (∀ m ω, IsProbabilityMeasure (h m ω)) ∧
    (∀ m ∈ box L β, ∀ m' ∈ box L β, ∀ ω,
      W1 (h m ω) (h m' ω) ≤ ENNReal.ofReal (Λ * msgDist m m')) ∧
    K = fun o ω => h (g o) ω}

/-- **prop:nofree, large side.** Padding the message with zeros and letting the receiver
ignore the extra values: every family of move laws available at size L' is available at
every size L ≥ L'. Hence any throughput that depends on the pair only through these laws
(no charge binding, messages always arriving) has a supremum nondecreasing in L. -/
def PaddingEmbedding : Prop :=
  ∀ (X O Ω : Type) [MetricSpace X] [MeasurableSpace X] (L' L : ℕ) (β Λ : ℝ),
    0 ≤ β → 0 ≤ Λ → L' ≤ L → AdmissibleLaws X O Ω L' β Λ ⊆ AdmissibleLaws X O Ω L β Λ

/-- **Witness bound, one time step** (note, Section "A larger message improves
throughput"): scaling a feasible point of (P1) at κ = 0 by θ = 1 − K/R gives a feasible point
with any charge 0 ≤ κ(i) ≤ K ≤ R. So the charged optimum is at least (1 − K/R) times every
uncharged value; with κ = 8 b(L)(1 + |𝒟_t(i)|) and 1 + |𝒟_t(i)| ≤ N one may take K = 8 N b(L). -/
def WitnessScaling : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V]
    (P : P1 V C D) (κ : D → ℝ) (K x : ℝ),
    0 < P.R → (∀ i, 0 ≤ κ i ∧ κ i ≤ K) → 0 ≤ K → K ≤ P.R →
    x ∈ P.values 0 → (1 - K / P.R) * x ∈ P.values κ

/-- The witness bound in expectation: if at every step and outcome the charged throughput
`thr` is at least θ times the uncharged one `u` (θ ≥ 0, from `WitnessScaling`) and the
uncharged episode throughput is at least τ_full − η, the charged one is at least
θ (τ_full − η) = τ₁. -/
def WitnessEpisode : Prop :=
  ∀ {Ξ : Type} [Fintype Ξ] (P : FinProb Ξ) (T : ℕ) (u thr : ℕ → Ξ → ℝ) (θ τfull η : ℝ),
    0 ≤ θ → (∀ t ξ, θ * u t ξ ≤ thr t ξ) →
    τfull - η ≤ (1 / T) * ∑ t ∈ Finset.range T, P.E (u t) →
    θ * (τfull - η) ≤ (1 / T) * ∑ t ∈ Finset.range T, P.E (thr t)

/-- **Corollary cor:air.** If at every step and outcome the charged throughput obeys
Proposition 1(a), thr ≤ τ⁰ − 8 b(L) Λ_t with Λ_t = ∑_i λ_t(i)(1 + |𝒟_t(i)|), and Φ is at
most the note's per-trajectory quantity τ_full − (1/T)∑_t τ⁰_t + 8 b(L)(1/T)∑_t Λ_t for every
outcome (the note's Φ(L), an infimum over trajectories, is the largest such Φ), then
τ̄ ≤ τ_full − Φ. -/
def CorAir : Prop :=
  ∀ {Ξ : Type} [Fintype Ξ] (P : FinProb Ξ) (T : ℕ) (thr τ0 Λsum : ℕ → Ξ → ℝ) (b τfull Φ : ℝ),
    (∀ t ξ, thr t ξ ≤ τ0 t ξ - 8 * b * Λsum t ξ) →
    (∀ ξ, Φ ≤ τfull - (1 / T) * ∑ t ∈ Finset.range T, τ0 t ξ
                + 8 * b * ((1 / T) * ∑ t ∈ Finset.range T, Λsum t ξ)) →
    (1 / T) * ∑ t ∈ Finset.range T, P.E (thr t) ≤ τfull - Φ

/-- **Corollary cor:middle, the combination step.** `A L` is the set of episode
throughputs reached by the pairs at size `L`, and τ*(L) = sup A(L). If every pair at L_min
is bounded by U_min < τ₁ (from thm:floor (iii) and (eq:small)), every pair at L_max by
U_max < τ₁ (from cor:air and Φ(L_max) > τ_full − τ₁), and some pair at L₁ reaches τ₁ (the
witness), then every maximizer of τ* over a set of tested sizes within [L_min, L_max]
that contains L₁ lies strictly between L_min and L_max. -/
def CorMiddle : Prop :=
  ∀ (A : ℕ → Set ℝ) (tested : Finset ℕ) (Lmin L1 Lmax : ℕ) (τ1 Umin Umax : ℝ),
    (∀ L ∈ tested, Lmin ≤ L ∧ L ≤ Lmax) → L1 ∈ tested →
    (A Lmin).Nonempty → (∀ x ∈ A Lmin, x ≤ Umin) → Umin < τ1 →
    (A Lmax).Nonempty → (∀ x ∈ A Lmax, x ≤ Umax) → Umax < τ1 →
    (∃ x ∈ A L1, τ1 ≤ x) → BddAbove (A L1) →
    ∀ L ∈ tested, (∀ L' ∈ tested, sSup (A L') ≤ sSup (A L)) → Lmin < L ∧ L < Lmax

end Madre
