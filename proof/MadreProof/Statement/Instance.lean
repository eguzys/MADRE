/-
# Proposition 1(a) at the simulator's constants: formal statement

Statements only (proofs in `MadreProof/Proofs/Instance.lean`). The constants are those of the
evaluated simulator, checked against the code (see the README correspondence table):

* R = 250 000 · 49/64 = 191 406.25 application bit/s per radio (`om2g/om2g_track.py`,
  `OM2G_RADIO_CAP_APP_BPS`: the 250 kbit/s 802.15.4 rate in the flows' own units, a 64-byte
  voice frame on air delivering 49 bytes), with one message round per 1 s step
  (`CONTROL_SYNC_INTERVAL_S = 1.0`), so κ(i) in bits per step equals κ(i) in bit/s;
* b(L), the on-air bytes of one message, for the seven tested sizes (paper, Section V);
* N = 8 drones (paper, Table I), so a drone has at most 7 neighbours.
-/
import MadreProof.Statement.NoFree

namespace Madre

/-- R, the per-drone radio budget of the evaluated LP, in application bit/s. -/
noncomputable def simR : ℝ := 250000 * 49 / 64

/-- N, the number of drones. -/
def simN : ℕ := 8

/-- The tested message sizes L. -/
def testedL : Finset ℕ := {8, 16, 32, 64, 128, 256, 512}

/-- b(L): on-air bytes of one message at the tested sizes (0 elsewhere, unused). -/
def onAirBytes : ℕ → ℝ
  | 8 => 131 | 16 => 198 | 32 => 262 | 64 => 460 | 128 => 786 | 256 => 1508 | 512 => 2882
  | _ => 0

/-- The floor factor 1 − 8 N b(L)/R of the witness bound. -/
noncomputable def floorFactor (L : ℕ) : ℝ := 1 - 8 * simN * onAirBytes L / simR

/-- **κ(i) ≤ R at every tested size** (the hypothesis of `LPOptimumExists`): with at most
N = 8 drones and 𝒟_t(i) ⊆ 𝒟 ∖ {i}, 0 ≤ κ(i) = 8 b(L)(1 + |𝒟_t(i)|) ≤ R. -/
def ChargeFitsRadio : Prop :=
  ∀ L ∈ testedL, ∀ {D : Type} [Fintype D] [DecidableEq D] (nbr : D → Finset D),
    Fintype.card D ≤ simN → (∀ i, i ∉ nbr i) →
    ∀ i, 0 ≤ charge (onAirBytes L) nbr i ∧ charge (onAirBytes L) nbr i ≤ simR

/-- **Two-sided airtime bracket at the simulator's constants.** For (P1) with R = simR,
nonnegative capacities and demand, at most 8 drones and 𝒟_t(i) ⊆ 𝒟 ∖ {i}, at every tested
size L: the optimal values τ_t (with the charge) and τ⁰_t (without) exist, and for every
optimal dual solution at κ = 0
  (1 − 8 N b(L)/R) τ⁰_t ≤ τ_t ≤ τ⁰_t − 8 b(L) ∑_i λ_t(i)(1 + |𝒟_t(i)|). -/
def AirtimeBracket : Prop :=
  ∀ {V C D : Type} [Fintype V] [Fintype C] [Fintype D] [DecidableEq V] [DecidableEq D]
    (P : P1 V C D) (L : ℕ) (nbr : D → Finset D),
    L ∈ testedL → P.R = simR → Fintype.card D ≤ simN → (∀ i, i ∉ nbr i) →
    (∀ v w, 0 ≤ P.c v w) → 0 ≤ P.demand →
    (∃ τ, IsGreatest (P.values (charge (onAirBytes L) nbr)) τ) ∧
    (∃ τ0, IsGreatest (P.values 0) τ0) ∧
    ∀ (τ τ0 : ℝ) (μ : V → V → ℝ) (ν : V → C → ℝ) (lam : D → ℝ) (ρ : C → ℝ),
      IsGreatest (P.values (charge (onAirBytes L) nbr)) τ →
      IsGreatest (P.values 0) τ0 →
      P.DualOptimal 0 μ ν lam ρ →
      floorFactor L * τ0 ≤ τ ∧
      τ ≤ τ0 - 8 * onAirBytes L * ∑ i, lam i * (1 + (nbr i).card)

/-- **The seven floor factors, exactly** (≈ 0.956, 0.934, 0.912, 0.846, 0.737, 0.496,
0.036 for L = 8, …, 512): at L = 64 the exchange costs at most 15.4 % of τ⁰_t at any
positions, at L = 512 up to 96.4 %. -/
def FloorValues : Prop :=
  floorFactor 8 = 732089 / 765625 ∧ floorFactor 16 = 714937 / 765625 ∧
  floorFactor 32 = 698553 / 765625 ∧ floorFactor 64 = 129573 / 153125 ∧
  floorFactor 128 = 564409 / 765625 ∧ floorFactor 256 = 379577 / 765625 ∧
  floorFactor 512 = 27833 / 765625

end Madre
