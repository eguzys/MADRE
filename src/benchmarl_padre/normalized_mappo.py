"""Normalized-advantage MAPPO — subclass of BenchMARL's Mappo.

BenchMARL hardcodes normalize_advantage=False inside Mappo._get_loss
(benchmarl/algorithms/mappo.py). This module provides a drop-in variant
with normalize_advantage=True without touching the installed library.

Value-target scale: with ~1000-step OMNeT episodes and per-step reward
~125, GAE value targets are O(1e4-1e5), and TorchRL 0.11's ClipPPOLoss/GAE
has no built-in value-target normalization, so a critic regressing raw
returns of that size does not learn.

This is handled by a single constant reward scale (REWARD_SCALE = 1e-4, see
src/benchmarl_padre/omnet_common.py's RewardScaling transform, applied at
env-collection time, upstream of GAE): reward ~125 * 1e-4 -> value targets
land at O(1-10) directly, with no running statistics, no forward hook, and
no critic-side code at all. See src/tests/test_reward_scale.py for the
assert that the scaled magnitude actually lands there.

Sigma floor: with a fixed entropy coefficient, a large value (0.03) keeps
sigma near its init value of ~1.0 (the entropy bonus outweighs the
policy-gradient term), while a smaller value (0.01) can let entropy collapse
to -4..-14 (sigma ~= 0.002-0.009 for a diagonal Normal, from
H = k*(0.5*log(2*pi*e) + log(sigma))). A single fixed entropy_coef cannot
serve both regimes. The standard remedy -- also used to keep entropy from
collapsing in
the reference SAC implementation (Haarnoja et al. 2018 clamp log-sigma to
[-20, 2]) -- is a floor on the policy standard deviation. TorchRL/tensordict
already exposes exactly this as a first-class, numerically-safe-by-default
parameter: `NormalParamExtractor(scale_lb=...)` (tensordict.nn.distributions,
default 1e-4, a numerical guard rather than an exploration floor).
BenchMARL's own Mappo._get_policy_for_loss constructs that extractor with
scale_mapping only, leaving scale_lb at the library default, which is not an
exploration floor. This module exposes scale_lb as a
config field and applies it to every NormalParamExtractor found in the
built policy -- see NormalizedMappo.get_policy_for_loss below.
"""
from dataclasses import dataclass, MISSING
from typing import Tuple

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.objectives import LossModule

from benchmarl.algorithms.mappo import Mappo, MappoConfig

# Recommended default for scale_lb (the minimum policy standard deviation
# NormalParamExtractor is allowed to output). This is a CONFIGURABLE
# HYPERPARAMETER meant to be ablated (see conf/algorithm/padre_mappo_norm*.yaml),
# not a tuned magic number -- 0.1 is a reasoned starting point only:
#   - scale_mapping="biased_softplus_1.0" outputs sigma=1.0 at zero
#     pre-activation, so a floor of 0.1 still permits a 10x sharpening of
#     the policy from initialisation (it does not freeze exploration).
#   - The observed entropy collapse reached sigma ~= 0.002-0.009 (see
#     derivation above); a floor of 0.1 blocks that collapse by roughly
#     1-2 orders of magnitude.
#   - The action space is [-1, 1]^2 and the simulator maps action magnitude
#     1.0 to a 20 m/step velocity (STEP_M in omnet_module/greedy_oracle.py),
#     so sigma_min=0.1 still leaves ~2 m/step of residual exploration noise
#     even once the policy has sharpened all the way down to the floor.
DEFAULT_SCALE_LB = 0.1


class NormalizedMappo(Mappo):
    """MAPPO with normalize_advantage=True (BenchMARL's Mappo hardcodes False)
    and a configurable floor on the policy standard deviation (scale_lb).
    """

    def __init__(self, **kwargs):
        # Pop scale_lb out BEFORE calling super().__init__ so this
        # NormalizedMappo-only kwarg never reaches Mappo/Algorithm.__init__
        # (which forwards **kwargs down into TorchRL/BenchMARL internals that
        # do not know about it and would raise a confusing TypeError there
        # instead of here).
        #
        # AlgorithmConfig.get_algorithm passes **self.__dict__, so a config
        # pickled BEFORE scale_lb existed (i.e. any pre-sigma-floor run's
        # config.pkl, which omnet_checkpoint.resume_experiment unpickles and
        # rebuilds from) has no scale_lb key at all. Refuse rather than
        # injecting a default: resuming a run under a policy-std
        # floor it was not trained with changes the algorithm mid-run, and a
        # bare KeyError here would not explain why.
        if "scale_lb" not in kwargs:
            raise RuntimeError(
                "NormalizedMappo: no 'scale_lb' in the algorithm config. This "
                "config predates the sigma floor (see the module docstring), "
                "so it is almost certainly an old run's config.pkl being "
                "resumed. Resuming it under a floor it never trained with "
                "would silently change the algorithm mid-run. Either start a "
                "fresh run (every conf/algorithm/padre_mappo_norm*.yaml now "
                f"sets scale_lb; recommended default {DEFAULT_SCALE_LB}), or "
                "set scale_lb explicitly on the unpickled algorithm_config "
                "and document that the resumed run changed hyperparameters."
            )
        self.scale_lb = kwargs.pop("scale_lb")
        super().__init__(**kwargs)

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        loss_module, recurrent = super()._get_loss(group, policy_for_loss, continuous)
        loss_module.normalize_advantage = True  # parent hardcodes False
        return loss_module, recurrent

    def get_policy_for_loss(self, group: str) -> TensorDictModule:
        """Build (or fetch the cached) policy, then apply the sigma floor.

        BenchMARL's Mappo builds the continuous-action policy as
        ProbabilisticActor(TensorDictSequential(actor_module, extractor_module))
        where extractor_module wraps a NormalParamExtractor constructed with
        scale_mapping only (benchmarl/algorithms/mappo.py:151), leaving
        scale_lb at the library's numerical-guard default of 1e-4. We walk
        the built module tree and overwrite scale_lb on every
        NormalParamExtractor found so the floor is actually enforced.

        Loud failure over silent no-op (house style, see comments in
        omnet_module/omnet_env.py): this checks that the exploration floor is
        actually wired in. If this ever finds zero extractors -- e.g. a
        discrete action space, or a future BenchMARL version that changes
        the actor architecture -- we raise instead of training on an
        unfloored policy.
        """
        policy = super().get_policy_for_loss(group)
        n_found = 0
        for module in policy.modules():
            if isinstance(module, NormalParamExtractor):
                module.scale_lb = self.scale_lb
                n_found += 1
        if n_found == 0:
            raise RuntimeError(
                f"NormalizedMappo.get_policy_for_loss({group!r}): found 0 "
                "NormalParamExtractor instances in the built policy module "
                "tree -- the scale_lb sigma floor was NOT applied. This "
                "happens for discrete action spaces (no NormalParamExtractor "
                "is built there) or if BenchMARL's actor architecture "
                "changes. Refusing to silently train without the floor."
            )
        return policy


@dataclass
class NormalizedMappoConfig(MappoConfig):
    """Config for NormalizedMappo — MappoConfig fields plus scale_lb.

    scale_lb (float): floor on the policy standard deviation output by
        NormalParamExtractor (see DEFAULT_SCALE_LB above for the derivation
        of the recommended 0.1 default). MISSING here, like every other
        field on this config, because NormalizedMappoConfig's schema uses
        MISSING sentinels -- every padre_mappo_norm*.yaml must set it
        explicitly.
    """

    scale_lb: float = MISSING

    @staticmethod
    def associated_class():
        return NormalizedMappo
