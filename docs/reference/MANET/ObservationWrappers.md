# ObservationWrappers.py

Observation Wrappers provide a neat and flexible way to define different observation spaces without needing to modify the environment class itself. This design enables efficient customization of observation spaces tailored to specific requirements.

The created observation wrappers account for the fact that while the full observation space —  considering node positions, node signal received, interference, user positions, and other metrics — can be helpful, it is often computationally costly. Therefore, the wrappers allow selective inclusion of observations to streamline training. However, the observation space in StableBaselines3 must have a constant size. This requirement motivates the distinction between fixed-sized and variable-sized observation spaces, both of which comply with this constraint:

- Fixed-sized observation spaces: These always maintain the size of the full observation space. This allows flexibility in including or excluding variables without altering the observation space size. It is particularly useful for retraining with previously excluded variables.
  
- Variable-sized observation spaces: These reduce the observation space to the minimum required size, including only the selected observations. While more efficient, they do not allow for easy expansion without redefining the space.

Both fixed-sized and variable-sized spaces maintain constant dimensions in practice, adhering to the requirements of StableBaselines3.

__Invariable Versions__

To handle symmetries in the observation space, invariable versions of these wrappers are introduced. Symmetry elimination ensures that the agent does not need to learn redundant states. For example, in a scenario where users $u_1$ and $u_2$ swap positions but exhibit identical signal power and behavior, the agent only needs to learn one of these states. The invariable versions sort observations based on positions, removing redundant symmetry.

__Observation Wrappers__

Currently, the following observation wrappers are available:

- [FS_COW](#MANET.ObservationWrappers.FS_COW): Fixed-sized configurable observation wrapper.
- [VS_COW](#MANET.ObservationWrappers.VS_COW): Variable-sized configurable observation wrapper.
- [FS_iCOW](#MANET.ObservationWrappers.FS_iCOW): Invariable fixed-sized configurable observation wrapper.
- [VS_iCOW](#MANET.ObservationWrappers.VS_COW): Invariable variable-sized configurable observation wrapper.

The full observation space is detailed below, offering comprehensive options, including positions for nodes and users, and customizable selections for signals, interference, and actions. While this space is robust, it can be computationally expensive. For each `ObservationWrapper` one can configure which observations to include with this obs_config dictionary to be handed to it:

```python
{
    "nodes": {
        "position": True,
        "receivedSignal": True,
        "interference": True,
        "signalPower": True,
        "trafficHosted": True,
        "inCapacities": True,
        "outCapacities": True,
        "sensedDemands": True,
        "inUserCount": True,
        "outUserCount": True,
        "closestUserDir": True,
        "closestUserInterference": True,
    },
    "users": {
        "position": True,
        "receivedSignal": True,
        "interference": True,
        "signalPower": True,
        "offeredLoad": True,
        "destination": True,
    },
    "capacity_matrix": True,
}
```

__Decentralized Execution and Partial State Loss__

The observation wrappers can be used to simulate PADRE's CTDE-based approach where a copy of the centralized agent is deployed on each MANET node individually. There is the `Base_ObservationWrapper` which facilitates collecting observations normally for a centralized agent and serves as the foundation for all observation wrappers. To simulate decentralized state collection and state loss, we use blocking observation wrappers that store previous component properties and block certain updates from being made, all inheriting from the `Blocking_Base_ObservationWrapper`.

Since state information is calculated from the properties of network components (MANET nodes and users), rather than directly blocking state information, these wrappers block updates of component properties, which is more realistic. The wrappers store the previous properties of network components from the last successful updates, not the calculated state itself.

The `Random_ObservationWrapper` is designed for centralized agents but simulates state update loss. For each component's property, a Bernoulli variable is sampled with update success probability of `1-state_loss_rate`. If an update fails, the old property value of that component will be used to calculate the current state.

The `Real_ObservationWrapper` is similar to the `Random_ObservationWrapper` with state loss but goes further. It is applied for each MANET node separately, where each node can only collect component property information from components that have a path to it. For each connected component, a Bernoulli distribution is applied to sample the success of an update, which occurs with probability `1-state_loss_rate`. For each MANET node a seperate RL agent needs to be wrapped (ideally, the same model), with the `node_idx` indicating the node it is deployed on.

For both wrappers, component properties are first initialized with all components placed at (0,0) without any network updates.

These can be used instead of the `Base_ObservationWrapper`, wrapping the typical observation wrappers (`FS_COW`, `FS_iCOW`, `VS_COW`, `VS_iCOW`) around them.


::: MANET.ObservationWrappers
    options:
      show_root_heading: false  # Avoid showing the module name as a header