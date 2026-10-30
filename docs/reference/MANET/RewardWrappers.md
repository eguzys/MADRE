# RewardWrappers.py

Reward Wrappers provide a neat and flexible way to define different reward functions without needing to modify the environment class itself. This design enables efficient customization of reward functions tailored to specific requirements.

__Reward Wrappers__

There are, so far, four reward wrappers:

- [MovementCost](#MANET.RewardWrappers.MovementCost): Penalizes the agent for moving, encouraging it to quickly move to the optimum position and then stop moving.
- [LinearReward](#MANET.RewardWrappers.LinearReward): Is simply the network throughput without modifications.
- [SquaredReward](#MANET.RewardWrappers.SquaredReward): Adjusts the reward based on the square of the network throughput, increasing the slope of the reward function.
- [EvaluationReward](#MANET.RewardWrappers.EvaluationReward): Provides a simple evaluation reward based on network throughput. This is practically the same as the `LinearReward` but is intended for evaluation purposes; thus, seperated.

::: MANET.RewardWrappers
    options:
      show_root_heading: false  # Avoid showing the module name as a header