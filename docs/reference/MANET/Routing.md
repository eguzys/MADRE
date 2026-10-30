# Routing.py

Routing holds multiple implementations of "Routing-Algorithms" that can be used for the network. Since the thesis focuses on perfect routing, these are not really routing algorithms in the traditional sense but rather "routing" as used in the MANET simulation.

## Implementations of the MCMF Problem

There are three implementations of the Multi-Commodity Min-Cost Flow (MCMF) problem:

### 1. Linear Programming-Based Implementations
These two implementations follow a similar formulation:

- [multi_commodity_max_flow_pulp_1](#MANET.Routing.MCMF.multi_commodity_max_flow_pulp_1)  
  Uses `PuLP` for solving the MCMF problem.

- [multi_commodity_max_flow_xpress](#MANET.Routing.MCMF.multi_commodity_max_flow_xpress)  
  Uses `Xpress` via `PuLP`.  
  Limitation: The XPress solver struggles with handling more than 500 constraints, making it impractical for larger networks.

### 2. Alternative `PuLP` Implementation
- [multi_commodity_max_flow_pulp_2](#MANET.Routing.MCMF.multi_commodity_max_flow_pulp_2)  
  Another formulation using `PuLP`.

### 3. Gurobi Implementation
- [multi_commodity_max_flow_gurobi](#MANET.Routing.MCMF.multi_commodity_max_flow_gurobi)  
  Uses `Gurobi` via `Pyomo`.

All MCMF implementations should be correct, but they tend to be rather slow in practice.

## Faster Max Flow-Based Implementation
Two simpler but faster approaches using only a single commodity max flow algorithm, which both have their downside.
- [sequential_max_flow](#MANET.Routing.MaxFlow.sequential_max_flow):  For each (source, sink, demand) tuple, the max flow algorithm is executed sequentially with flow limited by the demand, and edge capacities are reduced accordingly.
- [artificialartifical_src_sink_max_flow](#MANET.Routing.MaxFlow.artifical_src_sink_max_flow): Adding an artificial source node and an artificial sink node to then only execute a single commodity max flow algorithm.


::: MANET.Routing
    options:
      show_root_heading: false  # Avoid showing the module name as a header
