# Agents.py

To compare the trained Agents to some baseline models, [Agents.py](#MANET.Agents) was created. Here, multiple agents have been created, and one can simply do so too by inheriting from [AgentBase](#MANET.Agents.AgentBase).

Since all Agents use a sampling process, different sampling strategies have been created too. The agents themselves are not focused on what values to sample, rather how and how to react to the sampled values. So, the values to be sampled are interchangeable and that is made possible by using a [ConditionStrategy](ConditionStrategy.md)

__Agents__

There are, so far, five agents:

- [RandomAgent](#MANET.Agents.RandomAgent): Randomly chooses an action.
- [GreedyMaximizer](#MANET.Agents.GreedyMaximizer): Samples the values in all directions and moves for some steps in the direction with the best value.
- [OutAndThroughAgent](#MANET.Agents.OutAndThroughAgent): First moves away from noise or towards where any condition improves (out). If there is no large improvement, it moves in the other direction and checks there (through).
- [SpiralAgent](#MANET.Agents.SpiralSearch): Samples for the best position by moving in a spiral.
- [RandomSampler](#MANET.Agents.RandomSampler): Randomly chooses positions to sample and moves to the best one, including the ones taken while moving to those positions.

::: MANET.Agents
    options:
      show_root_heading: false  # Avoid showing the module name as a header
