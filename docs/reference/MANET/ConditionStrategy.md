# ConditionStrategy.py

The baseline agents in [Agents.py](Agents.md) share a common behavioral pattern: they sample environmental data to make positioning decisions, similar to how RL agents observe their environment state. However, these agents define *how* they sample and *what actions* they take, but not *what specific metrics* they should optimize for.

The ConditionStrategy classes provide this missing piece by implementing a modular system that defines what value to sample and when a position is considered viable. Each strategy includes a configurable threshold - when the sampled value reaches this threshold, the agent can stop moving and remain at its current position, conserving energy while maintaining acceptable performance.

This modular design allows agents to be easily reconfigured for different optimization objectives simply by swapping condition strategies, without modifying the agent's core sampling and movement behavior.

**Creating Custom Strategies**

To implement a custom condition strategy, inherit from one of the base classes depending on the scope of your metric:

- **LocalConditionStrategy** - For metrics measured at the individual node level
- **GlobalConditionStrategy** - For metrics measured at the network/environment level

**Available ConditionStrategies**

Local strategies (node-specific optimization):

- [SINR](#MANET.ConditionStrategy.SINRConditionStrategy): Maximizes the signal-to-interference-plus-noise ratio
- [Signal](#MANET.ConditionStrategy.SignalConditionStrategy): Maximizes received signal strength
- [Interference](#MANET.ConditionStrategy.InterferenceConditionStrategy): Minimizes interference levels
- [TrafficHosted](#MANET.ConditionStrategy.TrafficHostedConditionStrategy): Maximizes traffic volume handled by the node

Global strategies (network-wide optimization):

- [Throughput](#MANET.ConditionStrategy.ThroughputConditionStrategy): Maximizes overall network throughput

::: MANET.ConditionStrategy
    options:
      show_root_heading: false  # Avoid showing the module name as a header