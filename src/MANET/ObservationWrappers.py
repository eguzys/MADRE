from typing import Callable
from copy import deepcopy

# Default dictionary
from collections import defaultdict

# Standard stuff
import numpy as np

# Networkx to find graph nodes to have a path to specific node
import networkx as nx

# ObservationWrapper to customize the observation space
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium import spaces

from .Environments import (
    Basic_MANETEnv,
    Static_MANETEnv,
    Less_Static_MANETEnv,
    Dynamic_MANETEnv,
)
import MANET.Components as net_comps

"""
# ObservationWrappers.py

Observation Wrappers provide a neat and flexible way to define different observation spaces without needing to modify the environment class itself. This design enables efficient customization of observation spaces tailored to specific requirements.

The created observation wrappers account for the fact that while the full observation space—considering node positions, node signal received, interference, user positions, and other metrics—can be helpful, it is often computationally costly. Therefore, the wrappers allow selective inclusion of observations to streamline training. However, the observation space in StableBaselines3 must have a constant size. This requirement motivates the distinction between fixed-sized and variable-sized observation spaces, both of which comply with this constraint:

- **Fixed-sized observation spaces**: These always maintain the size of the full observation space. This allows flexibility in including or excluding variables without altering the observation space size. It is particularly useful for retraining with previously excluded variables.
  
- **Variable-sized observation spaces**: These reduce the observation space to the minimum required size, including only the selected observations. While more efficient, they do not allow for easy expansion without redefining the space.

Both fixed-sized and variable-sized spaces maintain constant dimensions in practice, adhering to the requirements of StableBaselines3.

### Invariable Versions

To handle symmetries in the observation space, invariable versions of these wrappers are introduced. Symmetry elimination ensures that the agent does not need to learn redundant states. For example, in a scenario where users $u_1$ and $u_2$ swap positions but exhibit identical signal power and behavior, the agent only needs to learn one of these states. The invariable versions sort observations based on positions, removing redundant symmetry.

## Observation Wrappers

Currently, the following observation wrappers are available:

- [FS_COW](#MANET.ObservationWrappers.FS_COW): Fixed-sized configurable observation wrapper.
- [VS_COW](#MANET.ObservationWrappers.VS_COW): Variable-sized configurable observation wrapper.
- [FS_iCOW](#MANET.ObservationWrappers.FS_iCOW): Invariable fixed-sized configurable observation wrapper.
- [VS_iCOW](#MANET.ObservationWrappers.VS_COW): Invariable variable-sized configurable observation wrapper.

The full observation space is detailed below, offering comprehensive options, including positions for nodes and users, and customizable selections for signals, interference, and actions. While this space is robust, it can be computationally expensive:

```python
{
    "nodes": {
        "position": True,
        "receivedSignal": False,
        "interference": True,
        "signalPower": True,
        "trafficHosted": False,
        "inCapacities": True,
        "outCapacities": True,
        "sensedSenderDemands": True,
        "sensedReceiverDemands": True,
        "inUserCount": True,
        "outUserCount": True,
        "inSenderCount": True,
        "outSenderCount": False,
        "inReceiverCount": False,
        "outReceiverCount": True,
        "closestUserDir": True,
        "closestUserPos": False,
        "closestSenderDir": True,
        "closestSenderPos": False,
        "closestReceiverDir": True,
        "closestReceiverPos": False,
        "closestUserInterference": True,
    },
    "users": {
        "position": False,
        "receivedSignal": False,
        "interference": False,
        "signalPower": False,
        "offeredLoad": False,
        "destination": False,
    },
    "capacity_matrix": False,
}
```
"""

DEFAULT_CONFIG = {
    "nodes": {
        "position": True,
        "receivedSignal": False,
        "interference": True,
        "signalPower": True,
        "trafficHosted": False,
        "inCapacities": True,
        "outCapacities": True,
        "sensedSenderDemands": True,
        "sensedReceiverDemands": True,
        "inUserCount": True,
        "outUserCount": True,
        "inSenderCount": True,
        "outSenderCount": False,
        "inReceiverCount": False,
        "outReceiverCount": True,
        "closestUserDir": True,
        "closestUserPos": False,
        "closestSenderDir": True,
        "closestSenderPos": False,
        "closestReceiverDir": True,
        "closestReceiverPos": False,
        "closestUserInterference": True,
    },
    "users": {
        "position": False,
        "receivedSignal": False,
        "interference": False,
        "signalPower": False,
        "offeredLoad": False,
        "destination": False,
    },
    "capacity_matrix": False,
}

NODE_KEYS = [
    "position",
    "receivedSignal",
    "interference",
    "signalPower",
    "trafficHosted",
    "inCapacities",
    "outCapacities",
    "sensedSenderDemands",
    "sensedReceiverDemands",
    "inUserCount",
    "outUserCount",
    "inSenderCount",
    "outSenderCount",
    "inReceiverCount",
    "outReceiverCount",
    "closestUserDir",
    "closestUserPos",
    "closestSenderDir",
    "closestSenderPos",
    "closestReceiverDir",
    "closestReceiverPos",
    "closestUserInterference",
]

USER_KEYS = [
    "position",
    "receivedSignal",
    "interference",
    "signalPower",
    "offeredLoad",
    "destination",
]

GENERAL_KEYS = ["capacity_matrix"]

# Set to False to restore the original PADRE behaviour for backwards compatibility.
FIX_BUGS = True

# When True, a blocked node's graph features show the last successfully-observed values (simple per-node history); when False, they show defaults (-1) instead.
lastOccurance = True

# when a node is not reachable we send the old information about this node. AOI will add a number to the observation how old this observation is.
# if AOI is 0 then the infomation is recent. and each update when an update didnt happen the number should be incremented
AOI = False

# Age of Information normalisation cap. PerDrone_OW divides its per-(i, j)
# age counter by this value to produce the AoI column in [0, 1]. The
# normalised value saturates at 1.0 once a peer has been silent for
# `INITIAL_AOI` env-steps — that's the "never received recently"
# regime. Reset puts every (i, j) counter at 0, not at INITIAL_AOI;
# INITIAL_AOI is the divisor / cap, not the post-reset value.
INITIAL_AOI = 1000.0


# when we use technologies like ST the transformer looses sorted values.
# because each features of a drone is an input for the ST and its permutation invariant
# the ST has no knoledge which of the nodes it controls
# we add a selfBit to the observation of each node to solve this problem. This selfBit is 1 if the node is controlled by the agent and 0 otherwise. 
selfBit = True




class Base_ObservationWrapper:
    """The base of all observation wrappers. This one handles all around collecting and calculating observations, with parameters to set that allow to pad non-used observations to keep a fixed size or not for a variablze sized observation vector. However, this is only a blueprint and doesn't yet follow Gymnasium standards.

    Attributes:
        env (gym.Env): The environment to wrap.
        config (dict): Configuration dictionary for the wrapper, which indicates which observations to include.
        unwrapped_env (Basic_MANETEnv): The unwrapped environment.
        is_padded (bool): Whether to pad not included observations. Defaults to False.
        is_invariable (bool): Whether to order the observations such that they are invariable about which User is which. Defaults to False.

    """

    def __init__(
        self,
        env: gym.Env,
        config: dict = None,
    ) -> None:
        """Initialize the Base Observation Wrapper

        Args:
            env (gym.Env): Environment to wrap
            config (dict, optional): Configuration dictates which observations to include in the observation space. Defaults to None.
        """
        # Set padded and ivariable
        self.is_padded = False
        self.is_invariable = False

        # Func dicts
        self.NODE_OBSERVATION_FUNCTIONS = {}
        self.USER_OBSERVATION_FUNCTIONS = {}
        self.GENERAL_OBSERVATION_FUNCTIONS = {}

        # Set env attributes
        self.env: gym.Env = env
        self.unwrapped_env: Basic_MANETEnv = self.env.unwrapped

        self.nodes = None
        self.users = None
        self.node_positions = None
        self.node_to_idx = None
        self.user_positions = None
        self.user_to_idx = None
        self.numbOfNodes = self.unwrapped_env.numbOfNodes
        self.initializeAllComponents()

        # Ensure valid config
        if config is None:
            config = DEFAULT_CONFIG
        else:
            try:
                config = filter_config(config)
                config = extend_with_missing_keys(config)
            except:
                print(f"Unable to resolve observation config. Must be very outdated")
                config = DEFAULT_CONFIG
        self.config = config

        # Set min max tuples for observation normalization
        self.initializeMinMaxTuples()
        self.initializeFuncDicts()

    def initializeFuncDicts(self) -> None:
        """Initialize the dicts that are iterated over in the observation gathering methods to
        use the correct functions to gather individual observations.
        """

        self.NODE_OBSERVATION_FUNCTIONS = {
            "position": self.get_node_position,
            "receivedSignal": self.get_node_signal_received,
            "interference": self.get_node_interference,
            "signalPower": self.get_node_signal_power,
            "trafficHosted": self.get_node_traffic_hosted,
            "inCapacities": self.get_node_in_capacity,
            "outCapacities": self.get_node_out_capacity,
            "sensedSenderDemands": self.get_node_sensed_sender_demands,
            "sensedReceiverDemands": self.get_node_sensed_receiver_demands,
            "inUserCount": self.get_node_user_in_count,
            "outUserCount": self.get_node_user_out_count,
            "inSenderCount": self.get_node_sender_in_count,
            "outSenderCount": self.get_node_sender_out_count,
            "inReceiverCount": self.get_node_receiver_in_count,
            "outReceiverCount": self.get_node_receiver_out_count,
            "closestUserDir": self.get_node_closest_user_dir,
            "closestUserPos": self.get_node_closest_user_position,
            "closestSenderDir": self.get_node_closest_sender_dir,
            "closestSenderPos": self.get_node_closest_sender_position,
            "closestReceiverDir": self.get_node_closest_receiver_dir,
            "closestReceiverPos": self.get_node_closest_receiver_position,
            "closestUserInterference": self.get_node_closest_user_interference,
        }

        self.USER_OBSERVATION_FUNCTIONS = {
            "position": self.get_user_position,
            "receivedSignal": self.get_user_signal_received,
            "interference": self.get_user_interference,
            "destination": self.get_user_dest_indices,
            "signalPower": self.get_user_signal_power,
            "offeredLoad": self.get_user_offered_load,
        }

        self.GENERAL_OBSERVATION_FUNCTIONS = {
            "capacity_matrix": self.get_capacity_matrix,
        }

    def initializeMinMaxTuples(self) -> None:
        """
        Initialize the (min_value, max_value) tuples used to normalize the observations to fit into [-1,1]
        """
        # Position is normed by the size of the playground
        playground_size = self.unwrapped_env.SIZE
        self.posMinMaxTuple = (0, playground_size)

        # Signal and interference are normed by cutting max signal or interference received in practice arond -30 dBm
        self.signalMinMaxTuple = (-100, -20)  # dBm
        self.interferenceMinMaxTuple = self.signalMinMaxTuple

        # Signal power is normed by dividing it by 1W, assumed to be maximum value
        self.signalPowMinMaxTuple = (0, 1)  # W

        # Offered Load norm factor
        self.offeredLoadMinMaxTuple = (0, self.unwrapped_env.MAX_OFFERED_LOAD)
        self.offeredLoadSumMinMaxTuple = (
            0,
            self.unwrapped_env.numbOfUsers / 2 * self.unwrapped_env.MAX_OFFERED_LOAD,
        )

        # Capacity is normed by dividing it by the maximum possible capacity:
        max_signal_received = self.unwrapped_env.nodeSignalPow_mW
        min_interference_received = 0
        env_noise = net_comps.ENVIRONMENT_NOISE

        nodes: np.ndarray[net_comps.MANETNode] = self.unwrapped_env.nodes
        max_bandwidth = max([node.BANDWIDTH for node in nodes])

        max_capacity = net_comps.shannonTheorem(
            signalReceived_mW=max_signal_received,
            interferenceReceived_mW=min_interference_received,
            noiseReceived_mW=env_noise,
            BANDWIDTH=max_bandwidth,
        )
        self.capacityMinMaxTuple = (0, max_capacity)
        self.capacitySumMinMaxTuple = (0, 2 * max_capacity)

        # Traffic hosted normalization
        self.trafficHostedMinMaxTuple = (0, max_capacity)

        # Maximum imaginable numb of users
        self.numbOfUsersMinMaxTuple = (0, 100)

    def get_observation(self) -> np.ndarray:
        """
        Returns the observations, which here is a combination of the nodes' observations,
        the users' observations and the general observations, such as the capacity matrix

        Returns:
            (np.ndarray): The observations.
        """

        self.updateComponents()

        node_obs: np.ndarray = self.get_selected_node_observations()
        user_obs: np.ndarray = self.get_selected_user_observations()
        general_obs: list[np.ndarray] = self.get_selected_general_observations()

        # Flatten
        node_obs = node_obs.flatten()
        user_obs = user_obs.flatten()
        general_obs: np.ndarray = (
            np.concatenate([arr.flatten() for arr in general_obs])
            if general_obs != []
            else np.array([], dtype=np.float32)
        )

        # Concatenate
        observations = np.concatenate(
            (node_obs.flatten(), user_obs.flatten(), general_obs.flatten())
        )
        return observations

    def initializeAllComponents(self) -> None:
        """Ensure that self.nodes and self.users reference the correct objects."""

        print("*** BASE initializeAllComponents called  we are in line 347  in function named initializeAllComponents***")

        self.nodes = self.unwrapped_env.nodes
        self.users = self.unwrapped_env.users
        self.node_positions = self.unwrapped_env.node_positions
        self.node_to_idx = self.unwrapped_env.node_to_idx
        self.user_positions = self.unwrapped_env.user_positions
        self.user_to_idx = self.unwrapped_env.user_to_idx

    def updateComponents(self) -> None:
        """Update components if necessary."""
        if self.hasEnvBeenReset():
            self.initializeAllComponents()

    def _get_capacity_matrix(self) -> np.ndarray:
        """Returns the capacity matrix to use for graph-derived observations.
        Subclasses may override to return a masked version."""
        return self.unwrapped_env.network.getCapacityMatrix()

    def hasEnvBeenReset(self) -> bool:
        """Whether the underyling environmnet has been reset or not.

        Returns:
            bool: True if the underyling env has been reset, meaning a new episode has startet, else False.
        """

        return self.unwrapped_env.ep_step == 0

    def setPadded(self, is_padded: bool) -> None:
        """
        Sets the padded status of the object.

        Args:
            is_padded (bool): A boolean value indicating whether the object
                              should be marked as padded (True) or not (False).

        Returns:
            None
        """
        self.is_padded = is_padded

    def setInvariable(self, is_invariable: bool) -> None:
        """
        Sets the invariable status of the object.

        Args:
            is_invariable (bool): A flag indicating whether the object should be marked as invariable.
        """
        self.is_invariable = is_invariable

    # * Node obs matrix
    def get_selected_node_observations(self) -> np.ndarray:
        """Collects all node related observations, given the config, which specifies which observations to include.
        If the observatidons should be padded then instead of not including them a fixed value array of same shape is included
        in its place.

        Returns:
            np.ndarray: Stacked observations of shape(numbOfNodes, x) where x is the number of observations, however e.g. the positions since 2D add +2 to x.
        """

        observations = []
        node_config: dict = self.config["nodes"]
        for key in NODE_KEYS:
            use_obs = node_config.get(key, False)
            if use_obs or self.is_padded:
                obs = self.NODE_OBSERVATION_FUNCTIONS[key](
                    use_replacement_value=(not use_obs)
                )
                observations.append(obs)

        return (
            np.hstack(observations)
            if observations != []
            else np.array([], dtype=np.float32)
        )

    # * User obs matrix
    def get_selected_user_observations(self) -> np.ndarray:
        """Collects all user related observations, given the config, which specifies which observations to include.
        If the observations should be padded then instead of not including them a fixed value array of same shape is included
        in its place. If the user' observations should be invariable, their observations are sorted by their positions. This
        helps avoid symmetries.

        Returns:
            np.ndarray: Stacked observations of shape(numbOfUsers, x) where x is the number of observations, however e.g. the positions since 2D add +2 to x.
        """

        observations = []
        user_config: dict = self.config["users"]
        for key in USER_KEYS:
            use_obs = user_config.get(key, False)
            if use_obs or self.is_padded:
                # We choose the destinations as positions instead of indices if the user observations are chosen as invariable.
                if key == "destination" and self.is_invariable:
                    obs = self.get_user_dest_position(
                        use_replacement_value=(not use_obs)
                    )
                else:
                    obs = self.USER_OBSERVATION_FUNCTIONS[key](
                        use_replacement_value=(not use_obs)
                    )

                observations.append(obs)
        observations = (
            np.hstack(observations)
            if observations != []
            else np.array([], dtype=np.float32)
        )

        # Sort the observations if to be invariable by the users' positions.
        if self.is_invariable and observations.size > 0:
            positions = self.USER_OBSERVATION_FUNCTIONS["position"](
                use_replacement_value=(not use_obs)
            )
            sorted_indices = np.lexsort((positions[:, 1], positions[:, 0]))
            observations = observations[sorted_indices, :]

        return observations

    # * Remaining obs matrix
    def get_selected_general_observations(self) -> list[np.ndarray]:
        """Collects all general observations, given the config, which specifies which observations to include, in a list.
        If the observations should be padded then instead of not including them a fixed value array of same shape is included
        in its place.

        Returns:
            list: Collected arrays of observations.
        """

        observations = []
        general_config: dict = self.config
        for key in GENERAL_KEYS:
            use_obs = general_config.get(key, False)
            if use_obs or self.is_padded:
                obs = self.GENERAL_OBSERVATION_FUNCTIONS[key](
                    use_replacement_value=(not use_obs)
                )
                observations.append(obs)

        return observations

    # * Node observations
    def get_node_position(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's position, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation
        """
        if use_replacement_value:
            return get_padded_array((self.numbOfNodes, 2), 0)
        else:
            return normalize_and_transform(self.node_positions, self.posMinMaxTuple)

    def get_node_signal_received(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's signals received, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            signals = np.array([node.signalReceived_mW for node in self.nodes]).reshape(
                -1, 1
            )

            epsilon = 1e-9
            transform_function = lambda x: 10 * np.log10(x + epsilon)
            return normalize_and_transform(
                signals, self.signalMinMaxTuple, transform_function
            )

    def get_node_interference(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's interferences, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            interferences = np.array(
                [node.interferenceReceived_mW for node in self.nodes]
            ).reshape(-1, 1)

            epsilon = 1e-9
            transform_function = lambda x: 10 * np.log10(x + epsilon)

            return normalize_and_transform(
                interferences,
                self.interferenceMinMaxTuple,
                transform_function,
            )

    def get_node_signal_power(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's signal powers, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            signalpower = np.array(
                [node.signalPower_mW for node in self.nodes]
            ).reshape(-1, 1)
            return normalize_and_transform(signalpower, self.signalPowMinMaxTuple)

    def get_node_traffic_hosted(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's traffic hosted, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            traffichosted = np.array(
                [node.trafficHosted for node in self.nodes]
            ).reshape(-1, 1)
            return normalize_and_transform(traffichosted, self.trafficHostedMinMaxTuple)

    def get_node_in_capacity(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the sum of the capacities from all incoming edges.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Sum of the capacities from all incoming edges
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self._get_capacity_matrix()
            in_capacities = np.sum(
                capacity_matrix[:, range(self.numbOfNodes)], axis=0
            ).reshape(self.numbOfNodes, 1)
            return normalize_and_transform(in_capacities, self.capacitySumMinMaxTuple)

    def get_node_out_capacity(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the sum of the capacities of all outgoing edges.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Sum of the capacities of all outgoing edges
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self._get_capacity_matrix()
            out_capacities = np.sum(
                capacity_matrix[range(self.numbOfNodes), :], axis=1
            ).reshape(self.numbOfNodes, 1)
            return normalize_and_transform(out_capacities, self.capacitySumMinMaxTuple)

    def get_node_sensed_sender_demands(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the sum of demands of all users that have an edge to it and actually have a destination

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Sum of demands of all users that have an edge to a node for each node
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get capacity matrix and user demands
            capacity_matrix = self._get_capacity_matrix()

            user_demand_vector = np.zeros((self.unwrapped_env.numbOfUsers, 1))
            for i, user in enumerate(self.users):
                user: net_comps.User
                user_demand_vector[i] += user.offeredLoad if user.hasDest() else 0

            # Initialize the demand array for each node (nodes only)
            node_sensed_demand = np.zeros(
                (self.numbOfNodes, 1)
            )  # Now explicitly (numNodes, 1)

            # Extract the submatrix where rows are users, columns are nodes
            user_rows = np.arange(
                self.numbOfNodes, self.numbOfNodes + self.unwrapped_env.numbOfUsers
            )  # User indices in matrix
            node_cols = np.arange(self.numbOfNodes)  # Node indices in matrix

            # Get the relevant part of the matrix (Users → Nodes)
            user_to_node_matrix = capacity_matrix[user_rows][
                :, node_cols
            ]  # Shape: (numUsers, numNodes)
            user_to_node_matrix = np.sign(user_to_node_matrix)

            # Matrix-vector multiplication to sum up the demands at each node, keeping the (numNodes, 1) shape
            node_sensed_demand = (user_to_node_matrix.T @ user_demand_vector).reshape(
                self.numbOfNodes, 1
            )  # (numNodes, 1)

            return normalize_and_transform(
                node_sensed_demand, self.offeredLoadSumMinMaxTuple
            )

    def get_node_sensed_receiver_demands(
        self, use_replacement_value: bool
    ) -> np.ndarray:
        """For each node returns the sum of demands that the users it has an edge to are supposed to receive. So, these are the destination users of the demand.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Sum of demands that the users it has an edge to are supposed to receive
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get capacity matrix and receiver demands
            capacity_matrix = self._get_capacity_matrix()

            receiver_demands = np.zeros(
                (self.unwrapped_env.numbOfUsers, 1)
            )  # Preallocate array with the correct shape

            for user in self.users:
                user: net_comps.User
                if user.hasDest():
                    dest_idx = self.user_to_idx[user.dest]
                    receiver_demands[
                        dest_idx
                    ] += user.offeredLoad  # Update directly in the preallocated array

            # Extract the submatrix where rows are nodes, columns are users
            node_rows = np.arange(self.numbOfNodes)  # Node indices in matrix
            user_cols = np.arange(
                self.numbOfNodes, self.numbOfNodes + self.unwrapped_env.numbOfUsers
            )  # User indices in matrix

            # Get the relevant part of the matrix (Users → Nodes)
            node_to_user_matrix = capacity_matrix[node_rows][
                :, user_cols
            ]  # Shape: (numUsers, numNodes)
            node_to_user_matrix = np.sign(node_to_user_matrix)

            # Matrix-vector multiplication to sum up the demands at each node, keeping the (numNodes, 1) shape
            node_sensed_receiver_demand = node_to_user_matrix @ receiver_demands

            return normalize_and_transform(
                node_sensed_receiver_demand, self.offeredLoadSumMinMaxTuple
            )

    def get_node_user_in_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the number of user that have an edge to it.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Number of user that have an edge to a node for each node
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get capacity matrix from the environment
            capacity_matrix = self._get_capacity_matrix()

            # Define row and column indices for users and nodes
            user_rows = np.arange(
                self.numbOfNodes, self.numbOfNodes + self.unwrapped_env.numbOfUsers
            )  # User indices in matrix
            node_cols = np.arange(self.numbOfNodes)  # Node indices in matrix

            # Extract relevant submatrices
            user_to_node_matrix = capacity_matrix[user_rows][
                :, node_cols
            ]  # Shape: (numUsers, numNodes)

            # Use np.sign() + np.sum() for optimized counting
            num_users_to_node = np.sum(np.sign(user_to_node_matrix), axis=0).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_to_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_user_out_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node return the number of users it has an edge to.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the number of users to which it has an edge
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self._get_capacity_matrix()
            # Define row and column indices for users and nodes
            node_rows = np.arange(self.numbOfNodes)  # Node indices in matrix
            user_cols = np.arange(
                self.numbOfNodes, self.numbOfNodes + self.unwrapped_env.numbOfUsers
            )  # User indices in matrix

            node_to_user_matrix = capacity_matrix[node_rows][
                :, user_cols
            ]  # Shape: (numNodes, numUsers)
            num_users_from_node = np.sum(np.sign(node_to_user_matrix), axis=1).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_from_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_sender_in_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the count of incoming edges from users that send, in other words, have a destination.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the count of incoming edges from users that send, in other words, have a destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get all sending user indices
            sending_users_indices = np.array(
                [self.user_to_idx[user] for user in self.users if user.hasDest()]
            )
            if sending_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 1))

            # Get capacity matrix from the environment
            capacity_matrix = self._get_capacity_matrix()

            # Define row and column indices for users and nodes
            user_rows = sending_users_indices + self.numbOfNodes
            node_cols = np.arange(self.numbOfNodes)  # Node indices in matrix

            # Extract relevant submatrices
            user_to_node_matrix = capacity_matrix[user_rows][
                :, node_cols
            ]  # Shape: (numUsers, numNodes)

            # Use np.sign() + np.sum() for optimized counting
            num_users_to_node = np.sum(np.sign(user_to_node_matrix), axis=0).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_to_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_sender_out_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the count of outgoing edges to users that send, in other words, have a destination.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the count of outgoing edges to users that send, in other words, have a destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self._get_capacity_matrix()

            # Get all sending user indices
            sending_users_indices = np.array(
                [self.user_to_idx[user] for user in self.users if user.hasDest()]
            )
            if sending_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 1))

            # Define row and column indices for users and nodes
            node_rows = np.arange(self.numbOfNodes)  # Node indices in matrix
            user_cols = sending_users_indices + self.numbOfNodes

            node_to_user_matrix = capacity_matrix[node_rows][
                :, user_cols
            ]  # Shape: (numNodes, numUsers)
            num_users_from_node = np.sum(np.sign(node_to_user_matrix), axis=1).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_from_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_receiver_in_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the count of incoming edges from users that receive, in other words, users that are the destination of another user's offered load.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the count of incoming edges from users that receive, in other words, users that are the destination of another user's offered load.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get all receiving user indices
            receiving_users_indices = np.array(
                [self.user_to_idx[user.dest] for user in self.users if user.hasDest()]
            )
            if receiving_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 1))
            # Get capacity matrix from the environment
            capacity_matrix = self._get_capacity_matrix()

            # Define row and column indices for users and nodes
            user_rows = receiving_users_indices + self.numbOfNodes
            node_cols = np.arange(self.numbOfNodes)  # Node indices in matrix

            # Extract relevant submatrices
            user_to_node_matrix = capacity_matrix[user_rows][
                :, node_cols
            ]  # Shape: (numUsers, numNodes)

            # Use np.sign() + np.sum() for optimized counting
            num_users_to_node = np.sum(np.sign(user_to_node_matrix), axis=0).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_to_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_receiver_out_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the count of outgoin edges to users that receive, in other words, users that are the destination of another user's offered load.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the count of outgoin edges to users that receive, in other words, users that are the destination of another user's offered load.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self._get_capacity_matrix()

            # Get all receiving user indices
            receiving_users_indices = np.array(
                [self.user_to_idx[user.dest] for user in self.users if user.hasDest()]
            )
            if receiving_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 1))

            # Define row and column indices for users and nodes
            node_rows = np.arange(self.numbOfNodes)  # Node indices in matrix
            user_cols = receiving_users_indices + self.numbOfNodes

            node_to_user_matrix = capacity_matrix[node_rows][
                :, user_cols
            ]  # Shape: (numNodes, numUsers)
            num_users_from_node = np.sum(np.sign(node_to_user_matrix), axis=1).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_from_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_closest_user_dir(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the normed direction to the closest user.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the normed direction to the closest user.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=-1)
        else:
            # Get node and user positions directly as NumPy arrays
            node_positions = self.node_positions
            user_positions = self.user_positions

            # Compute pairwise differences using broadcasting
            # Shape: (N, U, 2)
            differences = (
                user_positions[np.newaxis, :, :] - node_positions[:, np.newaxis, :]
            )

            # Compute distances (norms) for all pairs
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Extract directions to the closest users
            directions = differences[
                np.arange(self.numbOfNodes), closest_user_indices
            ]  # Shape: (N, 2)

            def transform_func(x: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(x, axis=1, keepdims=True)  # Shape: (N, 1)
                # Replace norms < 1 with 1
                adjusted_norms = (
                    np.maximum(norms, 1.0) + 1e-9
                )  # Small epsilon for numerical stability
                return (
                    x / adjusted_norms
                )  # Broadcasting divides each row by its adjusted norm

            # Apply transformation with default min_max_tuple=(-1, 1)
            return normalize_and_transform(
                directions, transform_function=transform_func
            )

    def get_node_closest_sender_dir(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the normed direction to the closest sending user, one that has a destination.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Normed direction to the closest sending user, one that has a destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)
        else:
            # Get node positions directly as NumPy array
            node_positions = self.node_positions

            # Identify sender users (those with a destination)
            sending_users_indices = np.array(
                [self.user_to_idx[user] for user in self.users if user.hasDest()]
            )
            if sending_users_indices.size == 0:
                return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)

            # Get positions of sender users
            user_positions = self.user_positions[sending_users_indices]

            # Compute pairwise differences using broadcasting
            # Shape: (N, S, 2) where S is number of sender users
            differences = (
                user_positions[np.newaxis, :, :] - node_positions[:, np.newaxis, :]
            )

            # Compute distances (norms) for all pairs
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, S)

            # Find the index of the closest sender user for each node
            closest_sender_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Extract directions to the closest sender users
            directions = differences[
                np.arange(self.numbOfNodes), closest_sender_indices
            ]  # Shape: (N, 2)

            def transform_func(x: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(x, axis=1, keepdims=True)  # Shape: (N, 1)
                # Replace norms < 1 with 1
                adjusted_norms = (
                    np.maximum(norms, 1.0) + 1e-9
                )  # Small epsilon for numerical stability
                return (
                    x / adjusted_norms
                )  # Broadcasting divides each row by its adjusted norm

            # Apply transformation with default min_max_tuple=(-1, 1)
            return normalize_and_transform(
                directions, transform_function=transform_func
            )

    def get_node_closest_receiver_dir(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the normed direction to the closest receiving user, one that is the destination of another users offered load.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Normed direction to the closest receiving user, one that is the destination of another users offered load.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)
        else:
            # Store node positions for easier reference
            node_positions = self.node_positions

            # Find unique indices of all receiver users (users that are destinations for other users)
            receiver_users_indices = list(
                {self.user_to_idx[user.dest] for user in self.users if user.hasDest()}
            )
            if len(receiver_users_indices) == 0:
                return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)

            # Extract positions of only the receiver users (without duplicates)
            receiver_positions = self.user_positions[receiver_users_indices]

            # Compute pairwise differences using broadcasting
            # Shape: (N, S, 2) where S is number of sender users
            differences = (
                receiver_positions[np.newaxis, :, :] - node_positions[:, np.newaxis, :]
            )

            # Compute distances (norms) for all pairs
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, S)

            # Find the index of the closest sender user for each node
            closest_sender_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Extract directions to the closest sender users
            directions = differences[
                np.arange(self.numbOfNodes), closest_sender_indices
            ]  # Shape: (N, 2)

            def transform_func(x: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(x, axis=1, keepdims=True)  # Shape: (N, 1)
                # Replace norms < 1 with 1
                adjusted_norms = (
                    np.maximum(norms, 1.0) + 1e-9
                )  # Small epsilon for numerical stability
                return (
                    x / adjusted_norms
                )  # Broadcasting divides each row by its adjusted norm

            # Apply transformation with default min_max_tuple=(-1, 1)
            return normalize_and_transform(
                directions, transform_function=transform_func
            )

    def get_node_closest_user_interference(
        self, use_replacement_value: bool
    ) -> np.ndarray:
        """For each node, returns the interference received from the closest user, normalized to [-1, 1].

        Args:
            use_replacement_value (bool): Whether to return a padded array instead of actual observations.

        Returns:
            np.ndarray: Normalized interference from the closest user for each node, shape (numbOfNodes, 1).
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get node and user positions directly as NumPy arrays
            node_positions = self.node_positions
            user_positions = self.user_positions

            # Compute pairwise distances using broadcasting
            # Shape: (N, 1, 2) - (1, U, 2) -> (N, U, 2)
            differences = (
                node_positions[:, np.newaxis, :] - user_positions[np.newaxis, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Get interference values from all users
            user_interferences = self.get_user_interference(False)

            # Extract interference from the closest user for each node
            closest_interferences = user_interferences[closest_user_indices, :]

            return closest_interferences

    def get_node_closest_user_position(self, use_replacement_value: bool) -> np.ndarray:
        """For each node, returns the position received from the closest user, normalized to [-1, 1]^2.

        Args:
            use_replacement_value (bool): Whether to return a padded array instead of actual observations.

        Returns:
            np.ndarray: Normalized position from the closest user for each node, shape (numbOfNodes, 2).
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)
        else:
            # Get node and user positions directly as NumPy arrays
            node_positions = self.node_positions
            user_positions = self.user_positions

            # Compute pairwise distances using broadcasting
            # Shape: (N, 1, 2) - (1, U, 2) -> (N, U, 2)
            differences = (
                node_positions[:, np.newaxis, :] - user_positions[np.newaxis, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Get interference values from all users
            user_positions = self.get_user_position(False)

            # Extract interference from the closest user for each node
            closest_positions = user_positions[closest_user_indices, :]

            return closest_positions

    def get_node_closest_sender_position(
        self, use_replacement_value: bool
    ) -> np.ndarray:
        """For each node, returns the position of the closest sending user, a user that has a destination.

        Args:
            use_replacement_value (bool): Whether to return a padded array instead of actual observations.

        Returns:
            np.ndarray: Position of the closest sending user, a user that has a destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=-1)
        else:
            # Get node positions directly as NumPy array
            node_positions = self.node_positions

            # Identify sender users (those with a destination)
            sending_users_indices = np.array(
                [self.user_to_idx[user] for user in self.users if user.hasDest()]
            )
            if sending_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 2))

            # Get positions of sender users
            user_positions = self.user_positions[sending_users_indices]

            # Compute pairwise distances using broadcasting
            # Shape: (N, 1, 2) - (1, U, 2) -> (N, U, 2)
            differences = (
                node_positions[:, np.newaxis, :] - user_positions[np.newaxis, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Get interference values from all users
            user_positions = self.get_user_position(False)

            # Extract interference from the closest user for each node
            closest_interferences = user_positions[closest_user_indices, :]

            return closest_interferences

    def get_node_closest_receiver_position(
        self, use_replacement_value: bool
    ) -> np.ndarray:
        """For each node, returns the position of the closest receiving user, a user that is another's destination.

        Args:
            use_replacement_value (bool): Whether to return a padded array instead of actual observations.

        Returns:
            np.ndarray: Position of the closest receiving user, a user that is another's destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=-1)
        else:
            # Store node positions for easier reference
            node_positions = self.node_positions

            # Find unique indices of all receiver users (users that are destinations for other users)
            receiver_users_indices = list(
                {self.user_to_idx[user.dest] for user in self.users if user.hasDest()}
            )
            if len(receiver_users_indices) == 0:
                return np.zeros(shape=(self.numbOfNodes, 2))

            # Extract positions of only the receiver users (without duplicates)
            receiver_positions = self.user_positions[receiver_users_indices]

            # Compute pairwise distances using broadcasting
            # Shape: (N, 1, 2) - (1, U, 2) -> (N, U, 2)
            differences = (
                node_positions[:, np.newaxis, :] - receiver_positions[np.newaxis, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Get interference values from all users
            user_positions = self.get_user_position(False)

            # Extract interference from the closest user for each node
            closest_interferences = user_positions[closest_user_indices, :]

            return closest_interferences

    # * User observations
    def get_user_position(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' positions, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array((self.unwrapped_env.numbOfUsers, 2), 0)
        else:
            user_positions = self.user_positions
            return normalize_and_transform(user_positions, self.posMinMaxTuple)

    def get_user_signal_received(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' signals received, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            signals = np.array([user.signalReceived_mW for user in self.users]).reshape(
                -1, 1
            )

            epsilon = 1e-9
            transform_function = lambda x: 10 * np.log10(x + epsilon)

            return normalize_and_transform(
                signals, self.signalMinMaxTuple, transform_function
            )

    def get_user_interference(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' interferences, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            interferences = np.array(
                [user.interferenceReceived_mW for user in self.users]
            ).reshape(-1, 1)

            epsilon = 1e-9
            transform_function = lambda x: 10 * np.log10(x + epsilon)

            return normalize_and_transform(
                interferences,
                self.interferenceMinMaxTuple,
                transform_function,
            )

    def get_user_dest_indices(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' destination indices, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            dest_indices = np.array(
                [
                    self.user_to_idx.get(user.dest, self.unwrapped_env.numbOfUsers)
                    for user in self.users
                ]
            ).reshape(-1, 1)
            return normalize_and_transform(
                dest_indices, (0, self.unwrapped_env.numbOfUsers)
            )

    def get_user_dest_position(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' destination positions, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array((self.unwrapped_env.numbOfUsers, 2), -1)
        else:
            dest_pos = np.array(
                [
                    (user.dest.pos if user.dest is not None else np.zeros((2,)))
                    for user in self.users
                ]
            )
            return normalize_and_transform(dest_pos, self.posMinMaxTuple)

    def get_user_signal_power(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' signal powers, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            signalpower = np.array(
                [user.signalPower_mW for user in self.users]
            ).reshape(-1, 1)
            return normalize_and_transform(signalpower, self.signalPowMinMaxTuple)

    def get_user_offered_load(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' offered loads, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            offeredload = np.array([user.offeredLoad for user in self.users]).reshape(
                -1, 1
            )
            return normalize_and_transform(offeredload, self.offeredLoadMinMaxTuple)

    # * General observations
    def get_capacity_matrix(self, use_replacement_value: bool) -> np.ndarray:
        """
        Returns the normalized capacity matrix.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: The normalized capacity matrix.
        """
        if use_replacement_value:
            return get_padded_array(
                (
                    self.numbOfNodes + self.unwrapped_env.numbOfUsers,
                    self.numbOfNodes + self.unwrapped_env.numbOfUsers,
                ),
                -1,
            )
        else:
            capacity_matrix = self._get_capacity_matrix()

        return normalize_and_transform(capacity_matrix, self.capacityMinMaxTuple)

    # Configure observation space size
    def get_observation_size(self) -> int:
        """Calculates the size of the observation space by querying observation functions.

        Returns:
            int: The total size of the observation space.
        """
        total_size = 0

        # Node observations
        for key in NODE_KEYS:
            if self.config["nodes"].get(key, False) or self.is_padded:
                obs_array = self.NODE_OBSERVATION_FUNCTIONS[key](
                    use_replacement_value=True
                )
                total_size += obs_array.size

        # User observations
        for key in USER_KEYS:
            if self.config["users"].get(key, False) or self.is_padded:
                if key == "destination" and self.is_invariable:
                    obs_array = self.get_user_dest_position(use_replacement_value=True)
                else:
                    obs_array = self.USER_OBSERVATION_FUNCTIONS[key](
                        use_replacement_value=True
                    )
                total_size += obs_array.size

        # General observations
        for key in GENERAL_KEYS:
            if self.config.get(key, False) or self.is_padded:
                obs_array = self.GENERAL_OBSERVATION_FUNCTIONS[key](
                    use_replacement_value=True
                )
                total_size += obs_array.size

        return total_size


class Blocking_Base_ObservationWrapper(Base_ObservationWrapper):
    """To simulate state loss and realistic observation updates in deployment of our rl-agent, the `Blocking_Base_ObservationWrapper` serves as a blueprint to allow one to customely define how observations are blocked/updated or not."""

    # * Override
    def updateComponents(self) -> None:
        """Update components if necessary."""
        if self.hasEnvBeenReset():
            self.initializeAllComponents()
        else:
            self.customBlockedUpdates()

    def customBlockedUpdates(self) -> None:
        """Custom update process that can be implemented e.g. by the blocking wrappers"""
        pass

    # * Overwrite
    def initializeAllComponents(self):
        """
        Updates all components by deep copying nodes and users to ensure independence,
        and shallow copying positions since they are numpy arrays of primitive types.
        """

        #print("*** BLOCKING initializeAllComponents called *** we are in class Blocking_Base_ObservationWrapper line 1495 in method initializeAllComponents. ")
        # Shallow copy positions since they are numpy arrays of primitive types
        self.node_positions = np.zeros(shape=(self.unwrapped_env.numbOfNodes, 2))
        self.user_positions = np.zeros(shape=(self.unwrapped_env.numbOfUsers, 2))

        # Deep copy nodes and users to ensure independence
        self.nodes: np.ndarray[net_comps.MANETNode] = np.array(
            [
                net_comps.MANETNode(
                    id=f"Node {i}",
                    pos=pos,
                    signalPower_mW=self.unwrapped_env.nodeSignalPow_mW,
                )
                for i, pos in enumerate(self.node_positions)
            ]
        )
        self.users: np.ndarray[net_comps.User] = np.array(
            [
                net_comps.User(
                    id=f"User {i}",
                    pos=pos,
                    signalPower_mW=self.unwrapped_env.userSignalPow_mW,
                )
                for i, pos in enumerate(self.user_positions)
            ]
        )

        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}
        self._masked_capacity_matrix = None

    def _get_capacity_matrix(self) -> np.ndarray:
        if FIX_BUGS and self._masked_capacity_matrix is not None:
            return self._masked_capacity_matrix
        return self.unwrapped_env.network.getCapacityMatrix()


class Real_ObservationWrapper(Blocking_Base_ObservationWrapper):
    """The `Real_ObservationWrapper` simulates realistic inter-node communication for exchanging the observation information. Further, it allows to probabilistically model data loss, through the `state_loss_rate` param. Each update fails with probability equal to the `state_loss_rate`. Assumes rl agent is deployed decentralised."""

    def __init__(
        self,
        node_idx: int,
        state_loss_rate: float,
        env: Basic_MANETEnv,
        config: dict = None,
    ):
        """Initialize.

        Args:
            node_idx (int): Id of the node it is deployed on. (in {0,..., numbOfNodes-1})
            state_loss_rate (float): Probability for component properties update to fail
            env (Basic_MANETEnv): The gymnasium environment which runs the scenario
            config (dict, optional): Config describing which observations to include. Defaults to None.
        """
        super().__init__(env, config)

        self.node_idx = node_idx

        assert (
            0 <= state_loss_rate <= 1
        ), "state_loss_rate must be between 0 and 1 (inclusive)."
        self.state_loss_rate = state_loss_rate

        self.np_random = np.random.default_rng(seed=103 + self.node_idx)

    # * Overwrite
    def customBlockedUpdates(self):
        """
        Updates self.nodes and self.users to include only the components (MANET nodes and users) that have a path to the current node (identified by self.node_index) in the graph. Further, for each update it randomly determines whether it succeeds. Each update fails with probability equal to the `state_loss_rate`.
        """
        # Get the current MANET node and the graph
        curr_node = self.unwrapped_env.nodes[self.node_idx]
        graph = self.unwrapped_env.network.getGraph()

        # Find all graph nodes that have a path to the current node
        connected_comps: list = sorted(
            nx.ancestors(graph, curr_node), key=lambda x: x.id
        )

        # Apply binomial distribution to filter connected components if state_loss_rate > 0
        if self.state_loss_rate > 0:
            mask = self.np_random.binomial(
                1, 1 - self.state_loss_rate, len(connected_comps)
            )
            connected_comps = [
                comp for comp, keep in zip(connected_comps, mask) if keep
            ]

        connected_comps.append(curr_node)  # Include the current node itself

        # Get the indices of the connected MANET nodes and users
        node_indices = [
            self.unwrapped_env.node_to_idx[comp]
            for comp in connected_comps
            if comp in self.unwrapped_env.node_to_idx
        ]
        user_indices = [
            self.unwrapped_env.user_to_idx[comp]
            for comp in connected_comps
            if comp in self.unwrapped_env.user_to_idx
        ]

        # Perform a deep copy of the nodes and users
        self.nodes[node_indices] = np.array(
            [deepcopy(self.unwrapped_env.nodes[idx]) for idx in node_indices]
        )
        self.users[user_indices] = np.array(
            [deepcopy(self.unwrapped_env.users[idx]) for idx in user_indices]
        )
        self.node_positions[node_indices] = np.copy(
            self.unwrapped_env.node_positions[node_indices]
        )
        self.user_positions[user_indices] = np.copy(
            self.unwrapped_env.user_positions[user_indices]
        )

        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}

        if FIX_BUGS:
            if lastOccurance:
                if self._masked_capacity_matrix is None:
                    size = self.numbOfNodes + self.unwrapped_env.numbOfUsers
                    self._masked_capacity_matrix = np.zeros((size, size))
                live = self.unwrapped_env.network.getCapacityMatrix()
                for i in node_indices:
                    self._masked_capacity_matrix[i, :] = live[i, :]
                    self._masked_capacity_matrix[:, i] = live[:, i]
                for u in user_indices:
                    u_mat = u + self.numbOfNodes
                    self._masked_capacity_matrix[u_mat, :] = live[u_mat, :]
                    self._masked_capacity_matrix[:, u_mat] = live[:, u_mat]
            else:
                connected_set = set(node_indices)
                blocked_node_indices = [i for i in range(self.numbOfNodes) if i not in connected_set]
                if blocked_node_indices:
                    live_matrix = self.unwrapped_env.network.getCapacityMatrix().copy()
                    for i in blocked_node_indices:
                        live_matrix[i, :] = 0.0
                        live_matrix[:, i] = 0.0
                    self._masked_capacity_matrix = live_matrix
                else:
                    self._masked_capacity_matrix = None


class PerDrone_OW(Blocking_Base_ObservationWrapper):
    """The `PerDrone_OW` produces N independent per-drone observations on
    each env step instead of a single global observation, where N equals
    `numbOfNodes`. For each drone i, drone j's row is populated iff there
    is a path from j to i in the connectivity graph AND a per-step
    Bernoulli(1 - state_loss_rate) draw succeeds for that link. Drone i's
    own row is always populated.

    Output of `get_observation()` has shape (N, N, n_features). Index
    [i, j, :] is drone j's state as seen by drone i. Missing/blocked
    entries follow the `Real_ObservationWrapper` default — zero on the
    first miss, last-seen values when FIX_BUGS and lastOccurance are
    enabled. The wrapper itself is stateless w.r.t. which drone is
    "active": every step it produces all N per-drone observations.

    Internally, all per-drone state lives in numpy buffers — a single
    cache of shape (N, N, n_features), per-drone masked capacity
    matrices, and per-drone RNGs. There are no per-drone deep-copied
    Python object lists; updates are numpy buffer writes.
    """

    def __init__(
        self,
        state_loss_rate: float,
        env: Basic_MANETEnv,
        config: dict = None,
        self_flag: bool = True,
        aoi: bool = True,
        history_K: int = 1,
        mask: bool = False,
    ):
        """Initialize.

        Args:
            state_loss_rate (float): Probability for a per-link state update to fail.
            env (Basic_MANETEnv): The gymnasium environment which runs the scenario.
            config (dict, optional): Config describing which observations to include. Defaults to None.
            self_flag (bool, optional): When True, append a binary self-flag
                column AFTER all feature columns. `out[i, i, self_flag_col] = 1.0`,
                `out[i, j, self_flag_col] = 0.0` for `i != j`. Permutation-equivariant
                policies (Set Transformer, GNN) can use this column to identify
                which row corresponds to "self" — necessary because such policies
                cannot rely on row order to encode that information. Defaults to True.
            aoi (bool, optional): When True, append an Age-of-Information column.
                Per-slot age normalised by INITIAL_AOI, capped at 1.0. With
                history_K > 1, each (i, j, k) slot tracks the age of the
                observation stored at that slot — NOT the slot index k. A k=0
                slot that holds a 5-step-old observation (peer was silent for 5
                steps) reports age 5/INITIAL_AOI, not 0. Defaults to True.
            history_K (int, optional): Number of historical observations kept per
                (i, j) link. Must be >= 1. K=1 (default) is backwards-compatible
                — cache layout collapses to the current per-step view. K>1
                packs the last K env-steps' observations as separate tokens.
                Output shape becomes (N, N*K, token_dim) when K > 1 (vs
                (N, N, token_dim) when K == 1).
            mask (bool, optional): When True, append a mask column at the very
                end (after aoi). 1.0 if the slot holds a real received
                observation since reset, 0.0 if filler (never received). The
                diagonal mask[i, i, 0] is True from reset; older diagonal slots
                flip True as the episode progresses. Defaults to False — the
                column is omitted to preserve bit-identity with the pre-history
                wrapper output at history_K=1.

        Column order in the per-(i, j, k) feature vector (last column is mask):
            [...feature columns..., self_flag (if on), aoi (if on), mask (if on)].
        """
        super().__init__(env, config)

        assert (
            0 <= state_loss_rate <= 1
        ), "state_loss_rate must be between 0 and 1 (inclusive)."
        assert history_K >= 1, "history_K must be >= 1."
        self.state_loss_rate = state_loss_rate
        self.self_flag = self_flag
        self.aoi = aoi
        self.history_K = int(history_K)
        self.mask = mask

        # Max N for pad-to-max buffer sizing (used in upcoming variable-N work).
        # Reads numbOfNodesRange[1] if env was configured with a range, else
        # falls back to scalar numbOfNodes. Currently UNUSED — plumbed only.
        _range = getattr(self.unwrapped_env, 'numbOfNodesRange', None)
        if isinstance(_range, (tuple, list)) and len(_range) == 2:
            self._max_numbOfNodes = int(_range[1])
        else:
            self._max_numbOfNodes = int(self.unwrapped_env.numbOfNodes)

        # Save the post-init buffers Blocking_Base_ObservationWrapper just
        # built. They serve as Real_ObservationWrapper's "default" view of
        # any never-reached peer (zero positions, zero attrs except
        # signalPower=nodeSignalPow_mW). Used to lazy-init the cache so
        # never-reached slots match Real_ObservationWrapper's output.
        self._default_nodes = self.nodes
        self._default_users = self.users
        self._default_node_positions = self.node_positions
        self._default_user_positions = self.user_positions
        self._default_node_to_idx = self.node_to_idx
        self._default_user_to_idx = self.user_to_idx

        # Per-drone RNGs — same seeding as Real_ObservationWrapper for
        # cross-implementation reproducibility.
        self._rngs: list[np.random.Generator] = [
            np.random.default_rng(seed=103 + i) for i in range(self._max_numbOfNodes)
        ]

        # Per-drone masked capacity matrix. Read by `_get_capacity_matrix`
        # only when `FIX_BUGS=True` and a drone is being actively
        # processed. Shape mirrors Blocking_Base_ObservationWrapper's
        # `_masked_capacity_matrix` so the same FIX_BUGS update rules apply.
        # Internal scratchpad — sized to CURRENT N+users, reallocated
        # on reset if either changes. Not seen by the NN, so fixed-shape
        # is unnecessary; matching live_capacity's shape is what matters.
        size = self.numbOfNodes + self.unwrapped_env.numbOfUsers
        self._per_drone_masked_capacity = np.zeros(
            (self.numbOfNodes, size, size)
        )

        # Cache layout per (i, j, k) slot (mask column lives outside cache):
        #   [0 : _n_features)                 — feature columns
        #   [_n_features]                     — self_flag (if on)
        #   [_n_features + self_flag_col]     — aoi (if on)
        # The self_flag column is set once below (identity across the
        # (i, j) plane, broadcast across k) and never written to again.
        # The aoi column is rewritten each step from `_aoi_counter`.
        # Feature updates target only `[:, :, :, :_n_features]`.
        self._n_features = self._compute_n_features_per_node()
        self._self_flag_col = self._n_features if self.self_flag else None
        self._aoi_col = (
            self._n_features + (1 if self.self_flag else 0)
            if self.aoi
            else None
        )
        cache_width = (
            self._n_features
            + (1 if self.self_flag else 0)
            + (1 if self.aoi else 0)
        )
        self._cache_width = cache_width
        self.cache = np.zeros(
            (self._max_numbOfNodes, self._max_numbOfNodes, self.history_K, cache_width),
            dtype=np.float32,
        )
        if self.self_flag and self._max_numbOfNodes > 0:
            # Broadcast identity across K: cache[i, i, k, self_flag_col] = 1.
            self.cache[:, :, :, self._self_flag_col] = np.eye(
                self._max_numbOfNodes, dtype=np.float32
            )[:, :, np.newaxis]

        # AoI counter (per slot): _aoi_counter[i, j, k] = env-steps since
        # the observation currently held in slot (i, j, k) was written.
        # Reset puts every counter at 0. The diagonal slot 0 stays at 0
        # (self is always rewritten each step). Older diagonal slots
        # accumulate age as the episode runs.
        if self.aoi and self._max_numbOfNodes > 0:
            self._aoi_counter = np.zeros(
                (self._max_numbOfNodes, self._max_numbOfNodes, self.history_K),
                dtype=np.float32,
            )

        # Mask buffer: True iff slot holds a real received observation.
        # On reset, the diagonal slot 0 is True (self always has its own
        # current state); everything else is False. Each receive shifts
        # the column back along K and sets slot 0 = True.
        if self._max_numbOfNodes > 0:
            self._mask_buffer = np.zeros(
                (self._max_numbOfNodes, self._max_numbOfNodes, self.history_K),
                dtype=bool,
            )
            # Diagonal write iterates current N (real drones only) —
            # absent drones' diagonals stay False.
            self._mask_buffer[np.arange(self.numbOfNodes),
                              np.arange(self.numbOfNodes), 0] = True

        # Active drone index — set during per-drone feature computation
        # so `_get_capacity_matrix` can return the right masked matrix.
        self._active_drone: int | None = None

        # Lazy init flag — defer cache initialization until the first
        # `get_observation()` call so the "default" features reflect the
        # env's state at that step (matching Real_ObservationWrapper's
        # behavior, which post-init buffers lookup against the live
        # capacity matrix at compute time).
        self._needs_cache_init = True

    def _compute_n_features_per_node(self) -> int:
        if self.numbOfNodes == 0:
            return 0
        node_config: dict = self.config["nodes"]
        total = 0
        for key in NODE_KEYS:
            if node_config.get(key, False) or self.is_padded:
                obs_array = self.NODE_OBSERVATION_FUNCTIONS[key](
                    use_replacement_value=True
                )
                total += obs_array.shape[1] if obs_array.ndim > 1 else 1
        return total

    # * Override
    def updateComponents(self) -> None:
        """Detect env reset and clear cache + per-drone capacity matrices.
        We do NOT call `customBlockedUpdates` here — per-drone updates
        are driven from `get_observation` so all drones share one pass.
        """
        super().updateComponents()
        if self.hasEnvBeenReset():
            # Refresh self.numbOfNodes from live env FIRST — Base wrapper
            # snapshots it once in __init__, so it's stale after a reset
            # that drew a different N. All sizing decisions below depend
            # on this value being current.
            self.numbOfNodes = self.unwrapped_env.numbOfNodes
            # Zero only the feature columns; the self-flag column (if any)
            # is constant identity across resets, and the aoi column is
            # rewritten from `_aoi_counter` each step.
            self.cache[:, :, :, : self._n_features].fill(0.0)
            new_size = self.numbOfNodes + self.unwrapped_env.numbOfUsers
            need_realloc = (
                self._per_drone_masked_capacity.shape[0] != self.numbOfNodes
                or self._per_drone_masked_capacity.shape[-1] != new_size
            )
            if need_realloc:
                self._per_drone_masked_capacity = np.zeros(
                    (self.numbOfNodes, new_size, new_size)
                )
            else:
                self._per_drone_masked_capacity.fill(0.0)
            # Refresh the default reference buffers from current env state.
            # (Original capture at PerDrone.__init__ assumes constant numbOfUsers.)
            self._wire_to_env()
            self._default_nodes = self.nodes
            self._default_users = self.users
            self._default_node_positions = self.node_positions
            self._default_user_positions = self.user_positions
            self._default_node_to_idx = self.node_to_idx
            self._default_user_to_idx = self.user_to_idx
            if self.aoi:
                self._aoi_counter.fill(0.0)
            if self.numbOfNodes > 0:
                self._mask_buffer.fill(False)
                self._mask_buffer[np.arange(self.numbOfNodes),
                                  np.arange(self.numbOfNodes), 0] = True
            self._needs_cache_init = True

    # * Override
    def _get_capacity_matrix(self) -> np.ndarray:
        """Returns the right capacity matrix for the active drone.
        - If `FIX_BUGS=True` and a drone is actively being processed,
          return that drone's masked matrix.
        - Otherwise return the env's live matrix. This matches the
          behavior `Blocking_Base_ObservationWrapper` provides via
          `_masked_capacity_matrix`, just keyed per active drone.
        """
        if FIX_BUGS and self._active_drone is not None:
            return self._per_drone_masked_capacity[self._active_drone]
        return self.unwrapped_env.network.getCapacityMatrix()

    def _wire_to_env(self) -> None:
        """Point self.* at env's live components (used during compute)."""
        self.nodes = self.unwrapped_env.nodes
        self.users = self.unwrapped_env.users
        self.node_positions = self.unwrapped_env.node_positions
        self.user_positions = self.unwrapped_env.user_positions
        self.node_to_idx = self.unwrapped_env.node_to_idx
        self.user_to_idx = self.unwrapped_env.user_to_idx

    def _wire_to_defaults(self) -> None:
        """Point self.* at the post-init default buffers (used to compute
        Real_ObservationWrapper's default view of a never-reached peer).
        """
        self.nodes = self._default_nodes
        self.users = self._default_users
        self.node_positions = self._default_node_positions
        self.user_positions = self._default_user_positions
        self.node_to_idx = self._default_node_to_idx
        self.user_to_idx = self._default_user_to_idx

    def _compute_default_features(self) -> np.ndarray:
        """Drone-agnostic "default" rows: post-init node/user buffers
        evaluated against the env's live matrix. Matches
        `Real_ObservationWrapper`'s output for never-reached peers when
        `FIX_BUGS=False`.
        """
        self._wire_to_defaults()
        try:
            return self.get_selected_node_observations()
        finally:
            self._wire_to_env()

    def _update_per_drone_masked_capacity(
        self,
        drone_idx: int,
        live_capacity: np.ndarray,
        reached_node_indices: list[int],
        reached_user_indices: list[int],
    ) -> None:
        """Update drone i's masked capacity matrix in place using the
        same rules as `Real_ObservationWrapper.customBlockedUpdates`."""
        if lastOccurance:
            for n in reached_node_indices:
                self._per_drone_masked_capacity[drone_idx, n, :] = live_capacity[n, :]
                self._per_drone_masked_capacity[drone_idx, :, n] = live_capacity[:, n]
            for u in reached_user_indices:
                u_mat = u + self.numbOfNodes
                self._per_drone_masked_capacity[drone_idx, u_mat, :] = live_capacity[u_mat, :]
                self._per_drone_masked_capacity[drone_idx, :, u_mat] = live_capacity[:, u_mat]
        else:
            reached_set = set(reached_node_indices)
            blocked = [n for n in range(self.numbOfNodes) if n not in reached_set]
            matrix = live_capacity.copy()
            for n in blocked:
                matrix[n, :] = 0.0
                matrix[:, n] = 0.0
            self._per_drone_masked_capacity[drone_idx] = matrix

    def get_observation(self) -> np.ndarray:
        """Per-step update of the (N, N, K, base_width) cache and emit
        the per-drone token tensor.

        On each call:
        1. Update self.cache for resets (handled in `updateComponents`).
        2. On the first call after reset, lazy-init every cache slot
           (broadcast across the K history dim) to the
           Real_ObservationWrapper "default" feature vector. Never-reached
           peer slots therefore match what Real_ObservationWrapper would
           emit.
        3. For each drone i, walk the env's connectivity graph, perform
           the per-link Bernoulli draw using i's RNG. For every reached
           j (including self), shift cache[i, j, 1:, :] back by one slot
           along K, then write the freshly observed features at slot 0.
           Mirror the same shift on the aoi counter and the mask buffer.
        4. Tick the aoi counter on all unwritten slots by +1 (so silent
           peers' slots age step by step).

        Output shape:
            (N, N,   token_dim) when history_K == 1
            (N, N*K, token_dim) when history_K > 1
        token_dim = features + self_flag (if on) + aoi (if on) + mask (if on)
        """
        self.updateComponents()
        self._wire_to_env()

        live_capacity = self.unwrapped_env.network.getCapacityMatrix()

        # Lazy init: fill cache with Real_ObservationWrapper "default" rows
        # at the current env state. Done once per reset so the matrix
        # used reflects the step at which observations are first read.
        if self._needs_cache_init:
            default_features = self._compute_default_features()  # (N_real, n_features)
            # Pad to max_N rows (zeros for absent drones) so broadcast
            # matches cache's max_N-sized peer dim.
            N_real = default_features.shape[0]
            max_N = self._max_numbOfNodes
            if N_real < max_N:
                pad = np.zeros((max_N - N_real, self._n_features), dtype=default_features.dtype)
                default_features = np.concatenate([default_features, pad], axis=0)
            # Broadcast across (drone i, slot k): cache[i, j, k, :_n_features]
            # = default_features[j, :].
            self.cache[:, :, :, : self._n_features] = default_features[
                np.newaxis, :, np.newaxis, :
            ]
            self._needs_cache_init = False

        graph = self.unwrapped_env.network.getGraph()

        # When FIX_BUGS=False, every drone sees the env's live capacity
        # matrix, so the feature matrix is identical across drones. Compute
        # it once and reuse — this is the main win over composition.
        shared_features = None
        if not FIX_BUGS:
            self._active_drone = None  # _get_capacity_matrix → env's live
            shared_features = self.get_selected_node_observations()

        K = self.history_K

        for i in range(self.numbOfNodes):
            curr_node = self.unwrapped_env.nodes[i]

            # Mirror Real_ObservationWrapper.customBlockedUpdates for
            # ancestor enumeration + per-link Bernoulli filtering.
            ancestors: list = sorted(
                nx.ancestors(graph, curr_node), key=lambda x: x.id
            )
            if self.state_loss_rate > 0 and ancestors:
                rng_mask = self._rngs[i].binomial(
                    1, 1 - self.state_loss_rate, len(ancestors)
                )
                reached = [a for a, k in zip(ancestors, rng_mask) if k]
            else:
                reached = list(ancestors)
            reached.append(curr_node)  # self always reached

            reached_node_indices = sorted(
                {
                    self.unwrapped_env.node_to_idx[c]
                    for c in reached
                    if c in self.unwrapped_env.node_to_idx
                }
            )
            reached_user_indices = sorted(
                {
                    self.unwrapped_env.user_to_idx[c]
                    for c in reached
                    if c in self.unwrapped_env.user_to_idx
                }
            )

            if FIX_BUGS:
                self._update_per_drone_masked_capacity(
                    i, live_capacity, reached_node_indices, reached_user_indices
                )
                self._active_drone = i
                try:
                    features_i = self.get_selected_node_observations()
                finally:
                    self._active_drone = None
            else:
                features_i = shared_features

            # Tick all slots in row i by +1 step. Reached slots get
            # overwritten below; unreached slots keep ageing.
            if self.aoi:
                self._aoi_counter[i, :, :] += 1.0

            if reached_node_indices:
                idx = np.asarray(reached_node_indices, dtype=np.int64)
                if K > 1:
                    # Shift older observations back along the K axis.
                    # Fancy indexing on the RHS returns a copy, so the
                    # in-place assignment is alias-safe.
                    self.cache[i, idx, 1:, : self._n_features] = (
                        self.cache[i, idx, :-1, : self._n_features]
                    )
                    self._mask_buffer[i, idx, 1:] = self._mask_buffer[i, idx, :-1]
                    if self.aoi:
                        self._aoi_counter[i, idx, 1:] = self._aoi_counter[i, idx, :-1]
                # Write the freshly observed features at slot 0.
                self.cache[i, idx, 0, : self._n_features] = features_i[idx, :]
                self._mask_buffer[i, idx, 0] = True
                if self.aoi:
                    self._aoi_counter[i, idx, 0] = 0.0

        if self.aoi:
            self.cache[:, :, :, self._aoi_col] = np.minimum(
                self._aoi_counter / INITIAL_AOI, 1.0
            )

        return self._emit_observation()

    def _emit_observation(self) -> np.ndarray:
        """Build the per-step output tensor from `self.cache` and (if
        `mask=True`) the mask buffer. Squeezes the K dim when K=1 to
        preserve backwards-compatible (max_N, max_N, token_dim) output.

        Output is always sized to `_max_numbOfNodes`; rows for absent
        drones (i >= current numbOfNodes) are zeroed entirely.
        """
        max_N = self._max_numbOfNodes
        N_real = self.numbOfNodes
        K = self.history_K
        if self.mask:
            mask_col = self._mask_buffer.astype(np.float32)[..., np.newaxis]
            out = np.concatenate([self.cache, mask_col], axis=-1)
        else:
            out = self.cache.copy()
        # Zero entire absent-drone rows. _mask_buffer.fill(False) on
        # reset already keeps mask col 0 for absent rows, but
        # self_flag's persistent eye[max_N] leaves diagonals at 1 —
        # must zero here explicitly.
        if N_real < max_N:
            out[N_real:, :, :, :] = 0.0
        token_dim = out.shape[-1]
        if K == 1:
            return out[:, :, 0, :]
        return out.reshape(max_N, max_N * K, token_dim)

    def get_observation_size(self) -> int:
        """Total flat size of the per-drone token tensor.
        Adds one column per enabled toggle (self_flag, aoi, mask) and
        multiplies the per-drone token count by history_K."""
        per_slot = (
            self._n_features
            + (1 if self.self_flag else 0)
            + (1 if self.aoi else 0)
            + (1 if self.mask else 0)
        )
        return self._max_numbOfNodes * self._max_numbOfNodes * self.history_K * per_slot


class Random_ObservationWrapper(Blocking_Base_ObservationWrapper):
    """Assumes rl-agent is deployed centralized. For each node or user's properties randomly determines, whether they are updated given the `state_loss_rate` param. Each update fails with probability equal to the `state_loss_rate`."""

    def __init__(
        self,
        state_loss_rate: float,
        env: Basic_MANETEnv,
        config: dict = None,
    ):
        """Initialize.

        Args:
            state_loss_rate (float): Probability for component properties update to fail
            env (Basic_MANETEnv): The gymnasium environment which runs the scenario
            config (dict, optional): Config describing which observations to include. Defaults to None.
        """
        super().__init__(env, config)

        assert (
            0 <= state_loss_rate <= 1
        ), "state_loss_rate must be between 0 and 1 (inclusive)."
        self.state_loss_rate = state_loss_rate

        self.np_random = np.random.default_rng(seed=42)

    # * Overwrite
    def customBlockedUpdates(self):
        """
        Updates self.nodes and self.users with states of nodes and users, using binomial distribution
        to simulate state loss rate, where indices are selected with p=1-state_loss_rate.
        """

        # Get all node and user indices
        node_indices = np.arange(self.unwrapped_env.numbOfNodes)
        user_indices = np.arange(self.unwrapped_env.numbOfUsers)

        # Apply binomial distribution to select indices to update
        node_update_mask = self.np_random.binomial(
            1, 1 - self.state_loss_rate, self.unwrapped_env.numbOfNodes
        )
        user_update_mask = self.np_random.binomial(
            1, 1 - self.state_loss_rate, self.unwrapped_env.numbOfUsers
        )

        # Filter indices to update
        node_indices_to_update = node_indices[node_update_mask == 1]
        user_indices_to_update = user_indices[user_update_mask == 1]

        # Perform a deep copy of the nodes and users
        self.nodes[node_indices_to_update] = np.array(
            [deepcopy(self.unwrapped_env.nodes[idx]) for idx in node_indices_to_update]
        )
        self.users[user_indices_to_update] = np.array(
            [deepcopy(self.unwrapped_env.users[idx]) for idx in user_indices_to_update]
        )

        # Shallow copy positions since they are numpy arrays of primitive types
        self.node_positions[node_indices_to_update] = np.copy(
            self.unwrapped_env.node_positions[node_indices_to_update]
        )
        self.user_positions[user_indices_to_update] = np.copy(
            self.unwrapped_env.user_positions[user_indices_to_update]
        )

        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}

        if FIX_BUGS:
            if lastOccurance:
                if self._masked_capacity_matrix is None:
                    size = self.numbOfNodes + self.unwrapped_env.numbOfUsers
                    self._masked_capacity_matrix = np.zeros((size, size))
                live = self.unwrapped_env.network.getCapacityMatrix()
                for i in node_indices_to_update:
                    self._masked_capacity_matrix[i, :] = live[i, :]
                    self._masked_capacity_matrix[:, i] = live[:, i]
                for u in user_indices_to_update:
                    u_mat = u + self.numbOfNodes
                    self._masked_capacity_matrix[u_mat, :] = live[u_mat, :]
                    self._masked_capacity_matrix[:, u_mat] = live[:, u_mat]
            else:
                blocked_node_indices = node_indices[node_update_mask == 0]
                if len(blocked_node_indices) > 0:
                    live_matrix = self.unwrapped_env.network.getCapacityMatrix().copy()
                    for i in blocked_node_indices:
                        live_matrix[i, :] = 0.0
                        live_matrix[:, i] = 0.0
                    self._masked_capacity_matrix = live_matrix
                else:
                    self._masked_capacity_matrix = None


# * Configurable Obswrappers
class Configurable_ObservationWrapper(ObservationWrapper):
    """Base class for configurable observation wrappers."""

    def __init__(
        self,
        wrapped_env: Base_ObservationWrapper,
        is_padded: bool = False,
        is_invariable: bool = False,
    ):
        """Transforms the any observation wrapper that inherits from the `Base_ObservationWrapper` into a Gymnasium compliant `ObservationWrapper` through wrapping around it.

        Args:
            wrapped_env (Base_ObservationWrapper): _description_
            is_padded (bool, optional): _description_. Defaults to False.
            is_invariable (bool, optional): _description_. Defaults to False.
        """
        assert isinstance(wrapped_env, Base_ObservationWrapper)

        super().__init__(wrapped_env.env)
        self.unwrapped_env: Basic_MANETEnv = wrapped_env.unwrapped_env
        self.env: gym.Env = wrapped_env.env
        self.wrapped_env: Base_ObservationWrapper = wrapped_env
        self.wrapped_env.setPadded(is_padded)
        self.wrapped_env.setInvariable(is_invariable)

        # Set the observation space
        numbOfObservations = self.wrapped_env.get_observation_size()
        observation_space = spaces.Box(
            low=-1, high=1, shape=(numbOfObservations,), dtype=np.float32
        )
        self.unwrapped_env.setObservationSpace(observation_space)

    def observation(self, obs=None) -> np.ndarray:
        """Return the observations.

        Args:
            obs (np.ndarray, optional): Given observation, that may be processed to. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        return self.wrapped_env.get_observation()

    def initializeComps(self) -> None:
        self.wrapped_env.initializeAllComponents()

    def updateComps(self) -> None:
        self.wrapped_env.updateComponents()


# * FixedSize_ConfigurableObservationWrapper
class FS_COW(Configurable_ObservationWrapper):
    """
    FixedSize_ConfigurableObservationWrapper customizes the observation space for a fixed size.
    This ensures that the observation space always has the same fixed size, regardless of which
    observations are included or excluded. Missing observations are padded with default values.

    Attributes:
        wrapped_env (Base_ObservationWrapper): The wrapped environment to customize.
    """

    def __init__(self, wrapped_env: Base_ObservationWrapper) -> None:
        """
        Initializes the FixedSize_ConfigurableObservationWrapper.

        Args:
            wrapped_env (Base_ObservationWrapper): The environment to wrap.
        """
        super().__init__(wrapped_env, True, False)


# * VariableSize_ConfigurableObservationWrapper
class VS_COW(Configurable_ObservationWrapper):
    """
    VariableSize_ConfigurableObservationWrapper customizes the observation space for a variable size.
    The observation space dynamically adjusts to include only the selected observations, resulting
    in a more compact representation.

    Attributes:
        wrapped_env (Base_ObservationWrapper): The wrapped environment to customize.
    """

    def __init__(self, wrapped_env: Base_ObservationWrapper) -> None:
        """
        Initializes the VariableSize_ConfigurableObservationWrapper.

        Args:
            wrapped_env (Base_ObservationWrapper): The environment to wrap.
        """
        super().__init__(wrapped_env, False, False)


# * invariable_FixedSize_ConfigurableObservationWrapper
class FS_iCOW(Configurable_ObservationWrapper):
    """
    invariable_FixedSize_ConfigurableObservationWrapper customizes the observation space for a fixed size
    while ensuring invariability. This means that the observation size is fixed, and user observations
    are cleared of symmetries by sorting them based on their positions.

    Attributes:
        wrapped_env (Base_ObservationWrapper): The wrapped environment to customize.
    """

    def __init__(self, wrapped_env: Base_ObservationWrapper) -> None:
        """
        Initializes the invariable_FixedSize_ConfigurableObservationWrapper.

        Args:
            wrapped_env (Base_ObservationWrapper): The environment to wrap.
        """
        super().__init__(wrapped_env, True, True)


# * invariable_VariableSize_ConfigurableObservationWrapper
class VS_iCOW(Configurable_ObservationWrapper):
    """
    invariable_VariableSize_ConfigurableObservationWrapper customizes the observation space for a variable size
    while ensuring invariability. The observation size dynamically adjusts to include only the selected observations,
    and user observations are cleared of symmetries by sorting them based on their positions.

    Attributes:
        wrapped_env (Base_ObservationWrapper): The wrapped environment to customize.
    """

    def __init__(self, wrapped_env: Base_ObservationWrapper) -> None:
        """
        Initializes the invariable_VariableSize_ConfigurableObservationWrapper.

        Args:
            wrapped_env (Base_ObservationWrapper): The environment to wrap.
        """
        super().__init__(wrapped_env, False, True)


def is_valid_config(config: dict) -> bool:
    """
    Checks if the given configuration is valid.

    Args:
        config (dict): The configuration dictionary to validate.

    Returns:
        is_valid (bool): True if the configuration is valid, False otherwise.
    """

    def keys_in_other_config(config: dict, other_config: dict) -> bool:
        for key in config:
            if key not in other_config:
                return False
            if type(config[key]) != type(other_config[key]):
                return False
            if isinstance(config[key], dict):
                if not keys_in_other_config(config[key], other_config[key]):
                    return False
        return True

    return keys_in_other_config(config, DEFAULT_CONFIG)


def filter_config(config: dict) -> dict:
    """
    Filters the given config to remove keys that do not belong to the default config.

    Args:
        config (dict): The custom configuration dictionary to filter.

    Returns:
        dict: The filtered configuration dictionary.
    """

    def recursive_filter(config: dict, default: dict) -> dict:
        filtered_config = {}
        for key, value in config.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    filtered_config[key] = recursive_filter(value, default[key])
                elif isinstance(value, bool):
                    filtered_config[key] = value
        return filtered_config

    return recursive_filter(config, DEFAULT_CONFIG)


def extend_with_missing_keys(config: dict) -> dict:
    """
    Ensures all keys from the default config are present in the given config.
    Missing keys (including nested ones) are added with the value False.

    Args:
        config (dict): The config to extend.

    Returns:
        dict: The extended config with missing keys filled with False.
    """
    default_config = DEFAULT_CONFIG

    def recursive_extend(config: dict, default: dict) -> None:
        for key, default_value in default.items():
            if key not in config:
                config[key] = False  # Add missing key with False
            elif isinstance(default_value, dict):
                # Ensure that the sub-config matches the structure of the default
                if not isinstance(config[key], dict):
                    config[key] = False  # Fallback to False if it's not a dict
                else:
                    # Recursively call on the sub-dictionaries to ensure they match
                    recursive_extend(config[key], default_value)

    # Create a copy of the original config to avoid modifying it directly
    extended_config = config.copy()
    recursive_extend(extended_config, default_config)

    return extended_config


def keys_match(dict1: dict, dict2: dict) -> bool:
    """
    Checks if the keys of two dictionaries match.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        keys_match (bool): True if the keys match, False otherwise.
    """
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not keys_match(dict1[key], dict2[key]):
                return False
    return True


def get_padded_array(shape: tuple[int, int], pad_value: float) -> np.ndarray:
    return np.full(shape, fill_value=pad_value, dtype=np.float32)


def normalize_and_transform(
    values: np.ndarray,
    min_max_tuple: tuple[float, float] = (-1, 1),
    transform_function: Callable[[np.ndarray], np.ndarray] = None,
) -> np.ndarray:
    """
    Normalizes and optionally transforms the input values.

    Args:
        values (np.ndarray): Values to normalize and transform.
        min_max_tuple (tuple[float, float], optional): Tuple with a minimum and maximum value for a certain observation, defining its bounds.
        transform_function (Callable[[np.ndarray], np.ndarray], optional): Function to transform the observations, as some - e.g. the interference - scale non-linearly. Defaults to None.

    Returns:
        np.ndarray: The normalized and transformed values.
    """

    if transform_function:
        transformed_values = transform_function(values)
    else:
        transformed_values = values

    min_value, max_value = min_max_tuple
    normed_values = normalize_values(transformed_values, min_value, max_value)

    np.clip(a=normed_values, a_min=-1, a_max=1, out=normed_values)

    return normed_values.astype(np.float32)


def normalize_values(
    values: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    """Normalizes the values, such that they are spread over [-1, 1]

    Args:
        values (np.ndarray): Values to normalize
        min_value (float): Minimum possible value, they can take.
        max_value (float): Maximum possible value, they can take.

    Returns:
        np.ndarray: Normalized values
    """

    normed_values = 2 * (values - min_value) / (max_value - min_value) - 1

    return normed_values
