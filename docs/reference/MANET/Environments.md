# Environments

The MANET environments simulate scenarios where users (e.g., smartphones) communicate through a Mobile Ad-hoc Network (MANET) backbone while malicious attackers attempt to disrupt communication by jamming MANET nodes. The goal for MANET nodes is to maximize network throughput despite adversarial interference.

**Architecture Overview**

All environments are custom Gymnasium environments built for reinforcement learning training. They follow a modular design where the core environment handles simulation logic, but observation and reward computation are delegated to separate wrapper classes. This design requires wrapping any environment with an [ObservationWrapper](ObservationWrappers.md) and a [RewardWrapper](RewardWrappers.md) before use.

**Core Functionality**

- **Component Management**: Handles initialization, distribution, and creation of network components (nodes, users, jammers)
- **Network Simulation**: Integrates with the [Network](Network.md) module for communication modeling
- **Movement Dynamics**: Manages node repositioning, user mobility, and attacker movement patterns
- **Traffic Modeling**: Simulates communication patterns and data flow requirements

**Environment Hierarchy**

Three environments provide increasing levels of dynamism, all inheriting from the foundational `Basic_MANETEnv`:

- **Static_MANETEnv**: The simplest concrete environment where users remain stationary with fixed communication patterns. Traffic destinations and offered loads are constant throughout each episode. Features an enhanced termination condition that ends episodes early when maximum theoretical throughput is achieved, making it ideal for scenarios focusing purely on node positioning optimization.

- **Less_Static_MANETEnv**: Introduces realistic user mobility through the Reference Point Group Mobility (RPGM) model, where users move slowly within clusters. Communication patterns remain static, providing a middle ground between fully static and dynamic scenarios.

- **Dynamic_MANETEnv**: The most realistic environment featuring both user mobility and dynamic traffic patterns. Communication destinations evolve according to a simplified Markov chain that favors conversation continuity, while offered loads follow a two-state Markov model distinguishing between small data (16Mb mean) and large data (300Mb mean) transmissions.

**Key Configuration Parameters**

- **networkConnectedAtStart**: Ensures initial network connectivity through a Monte Carlo distribution algorithm that places nodes at user cluster centers
- **jammersSpawnNextToUsers**: Controls jammer placement strategy - either near randomly selected users or coordinated around a central attack point  
- **steps_till_jammers_active**: Delay before jammers begin interfering, allowing initial network establishment
- **step_jammers_start_moving**: Separate control for when jammers begin repositioning (for moving attacker models)
- **allow_early_ep_finish**: Enables intelligent episode termination when no progress is being made or optimal throughput is achieved

**Component Distribution**

The environment supports two distribution strategies:

*Connected Distribution*: Uses a Monte Carlo algorithm ensuring network connectivity. User clusters are formed using exponential and Dirichlet distributions, with nodes strategically placed at cluster centers and bridge positions to maintain graph connectivity.

*Random Distribution*: Components are distributed without connectivity constraints using normal distributions around playground center (nodes) or cluster-based distributions (users), with jammers placed either near users or in coordinated attack formations.

**Advanced Features**

- **Flexible Component Counts**: Supports both fixed numbers and interval ranges for users and jammers, with uniform random sampling per episode
- **Multiple Attacker Models**: Can randomly select from lists of attacker strategies each episode for varied training scenarios  
- **Early Termination Logic**: Monitors throughput changes and node movement to detect convergence or stagnation
- **Visualization Support**: Integrated GUI rendering with threaded execution for real-time monitoring

**Technical Implementation**

Each environment maintains separate random generators for different subsystems (nodes, jammers, users, network, attackers) ensuring reproducible results while allowing independent control over different sources of randomness. The playground operates on a $[0,$ `SIZE` $]^2$ coordinate system with `SIZE`=100 by default.

::: MANET.Environments
    options:
      show_root_heading: false