import MadreProof.Proofs.StrongDuality
import MadreProof.Proofs.Information
import MadreProof.Proofs.CommonOmega
import MadreProof.Proofs.Throughput
import MadreProof.Proofs.NoFree
import MadreProof.Proofs.Sanity
import MadreProof.Proofs.Instance
import MadreProof.Proofs.Deterministic
import MadreProof.Proofs.Receivers
import MadreProof.Proofs.ThroughputGeneral
import MadreProof.Proofs.EdgeSet
import MadreProof.Proofs.Threshold
import MadreProof.Proofs.Realized
import MadreProof.Proofs.WeightedAirtime
import MadreProof.Proofs.WeightedStrongDuality
import MadreProof.Proofs.WeightedInstance
import MadreProof.Proofs.WeightedEdgeSet
import MadreProof.Proofs.WeightedSanity

/-! Every main theorem, exactly as stated in the statement files, and the axioms it uses.
Only Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`) may appear.
`lake build` prints the output of the `#print axioms` commands below. -/

/-- Proposition 1 (a) and (b) together, exactly as stated in `MadreProof/Statement.lean`. -/
theorem Madre.proposition1 : Madre.PropositionA ∧ Madre.PropositionB :=
  ⟨Madre.propositionA, Madre.propositionB⟩

#print axioms Madre.propositionA
#print axioms Madre.propositionB
#print axioms Madre.proposition1
#print axioms Madre.complementarySlackness
#print axioms Madre.lpStrongDuality
#print axioms Madre.lpOptimumExists
#print axioms Madre.propositionB_commonOmega
#print axioms Madre.performanceDifference
#print axioms Madre.theoremFloorII
#print axioms Madre.theoremFloorIII
#print axioms Madre.receiverFiniteSupport
#print axioms Madre.chargeAntitone
#print axioms Madre.chargeAntitoneInL
#print axioms Madre.paddingEmbedding
#print axioms Madre.witnessScaling
#print axioms Madre.witnessEpisode
#print axioms Madre.corAir
#print axioms Madre.corMiddle
#print axioms Madre.propositionA_nonvacuous
#print axioms Madre.propositionB_receiver_nonvacuous
#print axioms Madre.chargeFitsRadio
#print axioms Madre.airtimeBracket
#print axioms Madre.floorValues
#print axioms Madre.propositionBDeterministic
#print axioms Madre.diracAnsweredWellIff
#print axioms Madre.tanhNormalLipschitz
#print axioms Madre.tanhNormalSatisfiesB
#print axioms Madre.performanceDifferenceGeneral
#print axioms Madre.theoremFloorIIIGeneral
#print axioms Madre.markovOfKernel
#print axioms Madre.edgeSetValuesEq
#print axioms Madre.propositionAEdgeSet
#print axioms Madre.unitWeightSpecialCase
#print axioms Madre.lpStrongDualityW
#print axioms Madre.lpOptimumExistsW
#print axioms Madre.propositionAW
#print axioms Madre.complementarySlacknessW
#print axioms Madre.witnessScalingW
#print axioms Madre.chargeAntitoneW
#print axioms Madre.airtimeBracketW
#print axioms Madre.edgeSetValuesEqW
#print axioms Madre.propositionAEdgeSetW
#print axioms Madre.propositionA_fromWeighted
#print axioms Madre.lpStrongDuality_fromWeighted
#print axioms Madre.complementarySlackness_fromWeighted
#print axioms Madre.airtimeBracket_fromWeighted
#print axioms Madre.propositionAW_nonvacuous
#print axioms Madre.realizedSatisfiesB
#print axioms Madre.realizedDeterministicLipschitz
#print axioms Madre.packingBound
#print axioms Madre.thresholdLemma
#print axioms Madre.countBoundNeverBinds
#print axioms Madre.nonAttainmentForcesEllZero
#print axioms Madre.nonAttainmentForcesEllZeroGeneral
#print axioms Madre.countBoundExcludesNothing
