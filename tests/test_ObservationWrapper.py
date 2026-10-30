# Need for testing
from unittest.mock import patch, MagicMock, Mock
import pytest

# Standard libraries
import numpy as np

# Type hinting
from typing import Tuple, Callable, Union, Optional
import numpy.typing as npt

# Gymnasium
import gymnasium as gym

# Own project libraries
import MANET.Components as comps
import MANET.Network as net
import MANET.Environments as envs
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw

# Utils
import Utils.training_utils as ut

# * Test returned observation type and values are within boundaries -----------------------------------------------------------------------------------------------


def create_wrapped_env(
    numbOfNodes: int = 5, numbOfJammers: int = 2, numbOfUsers: int = 10
) -> gym.Env:
    env = get_env(numbOfNodes, numbOfJammers, numbOfUsers)
    env = ut.wrapEnv(
        env=env, rew_wrapper=rw.SquaredReward, obs_wrapper=ow.VS_COW, use_monitor=False
    )
    return env


def get_env(
    numbOfNodes: int = 5, numbOfJammers: int = 2, numbOfUsers: int = 10
) -> envs.Basic_MANETEnv:
    env = envs.Basic_MANETEnv(
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        render_mode=None,
        seed=42,
    )
    env.reset(seed=42)

    return env


@pytest.fixture
def wrapped_env() -> gym.Env:
    env = create_wrapped_env()

    return env


@pytest.fixture
def wrapped_uninit_env() -> gym.Env:
    with patch("MANET.Environments.Basic_MANETEnv.reset") as reset_mock:
        reset_mock.return_value = (
            None,
            None,
        )  # Mock reset to prevent full initialization

        # Initialize the environment but without `reset` running fully
        env = (
            create_wrapped_env()
        )  # Or use `create_non_wrapped_env` if you want an unwrapped environment

        return env


class Test_TypesAndBoundaries:
    def test_type(self, wrapped_env):
        env: gym.Env = wrapped_env
        env.reset(seed=42)
        observation = env.get_wrapper_attr("observation")()
        assert isinstance(observation, np.ndarray)

    @pytest.mark.parametrize(
        "numbOfNodes, numbOfJammers, numbOfUsers",
        [(3, 1, 3), (4, 2, 5), (5, 3, 7), (6, 2, 8), (7, 3, 10)],
    )
    def test_boundaries_over_multiple_steps(
        self, numbOfNodes: int, numbOfJammers: int, numbOfUsers: int
    ):
        """Check whether observation space is normalized correctly such that all observations are in [-1,1]

        Args:
            numbOfNodes (int): the number of nodes in the scenario
            numbOfJammers (int): the jammers of nodes in the scenario
            numbOfUsers (int): the users of nodes in the scenario
        """
        env = create_wrapped_env(numbOfNodes, numbOfJammers, numbOfUsers)
        obs, _ = env.reset(seed=42)

        total_timesteps = 500
        for _ in range(total_timesteps):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)

            in_bounds(obs, -1, 1)

            if done or trunc:
                env.reset()


def in_bounds(array: np.ndarray, low_bound: float, up_bound: float) -> bool:
    """Checks if an array's values are each within the specified bounds

    Args:
        array (np.ndarray): array to check
        low_bound (float): lower bound
        up_bound (float): upper bound

    Returns:
        bool: whether the array is in the bound
    """
    in_bounds = np.all((array >= low_bound) & (array <= up_bound))

    return in_bounds


# * Test values returned by each observation function are correct ----------------------------------------------------------------------------


@pytest.fixture
def setUpEnv() -> Tuple[
    ow.Base_ObservationWrapper,
    net.Network,
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
]:
    env = get_env(2, 1, 2)
    wrapped_env: ow.Base_ObservationWrapper = ut.wrapWithBaseWrapper(
        env=env, obs_config=None, obs_type="full"
    )
    wrapped_env: ow.Configurable_ObservationWrapper = (
        ow.Configurable_ObservationWrapper(wrapped_env=wrapped_env)
    )
    network = env.network
    nodes = env.nodes
    for node in nodes:
        node: comps.MANETNode
        node.radius = 30
    users = env.users

    return wrapped_env, network, nodes, users


def updateEnv(env: ow.Configurable_ObservationWrapper) -> None:
    numbOfNodes = env.get_wrapper_attr("numbOfNodes")
    action = np.zeros((numbOfNodes, 2))
    _, _, _, _, _ = env.step(action)

    env.initializeComps()


class Test_ObservationValues:
    # Mock function that simply returns the input values
    def mock_normalize_and_transform(
        self,
        values: npt.NDArray[np.float32],
        min_max_tuple: Optional[Tuple[float, float]] = None,
        transform_function: Optional[
            Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
        ] = None,
    ) -> npt.NDArray[np.float32]:
        return values

    # Example test using the mock
    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_position(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_node_positions = np.array([node.pos for node in nodes])

        res_node_positions = base_wrapped_env.get_node_position(False)

        assert np.array_equal(exp_node_positions, res_node_positions)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_signal_received(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_node_signals = np.array([node.signalReceived_mW for node in nodes]).reshape(
            -1, 1
        )

        res_node_signals = base_wrapped_env.get_node_signal_received(False)

        assert np.array_equal(exp_node_signals, res_node_signals)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_interference(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_node_interferences = np.array(
            [node.interferenceReceived_mW for node in nodes]
        ).reshape(-1, 1)

        res_node_interferences = base_wrapped_env.get_node_interference(False)

        assert np.array_equal(exp_node_interferences, res_node_interferences)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_signal_power(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_node_signalPow = np.array([node.signalPower_mW for node in nodes]).reshape(
            -1, 1
        )

        res_node_signalPow = base_wrapped_env.get_node_signal_power(False)

        assert np.array_equal(exp_node_signalPow, res_node_signalPow)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_traffic_hosted(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_node_traffic_hosted = np.array(
            [node.trafficHosted for node in nodes]
        ).reshape(-1, 1)

        res_node_traffic_hosted = base_wrapped_env.get_node_traffic_hosted(False)

        assert np.array_equal(exp_node_traffic_hosted, res_node_traffic_hosted)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_in_capacity(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env
        network: net.Network

        updateEnv(wrapped_env)

        # Get in capacities
        graph = network.getGraph()
        incoming_capacities = []

        for node in nodes:
            # Get all incoming edges to this node
            in_edges = graph.in_edges(node, data=True)
            # Sum the capacities of all incoming edges
            total_capacity = sum(edge[2]["capacity"] for edge in in_edges)

            incoming_capacities.append(total_capacity)

        exp_node_in_capacities = np.array(incoming_capacities).reshape(-1, 1)
        res_node_in_capacities = base_wrapped_env.get_node_in_capacity(False)

        assert np.array_equal(exp_node_in_capacities, res_node_in_capacities)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_out_capacity(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env
        network: net.Network

        updateEnv(wrapped_env)

        # Get in capacities
        graph = network.getGraph()
        outcoming_capacities = []

        for node in nodes:
            # Get all outcoming edges to this node
            out_edges = graph.out_edges(node, data=True)
            # Sum the capacities of all outcoming edges
            total_capacity = sum(edge[2]["capacity"] for edge in out_edges)

            outcoming_capacities.append(total_capacity)

        exp_node_out_capacities = np.array(outcoming_capacities).reshape(-1, 1)
        res_node_out_capacities = base_wrapped_env.get_node_out_capacity(False)

        assert np.array_equal(exp_node_out_capacities, res_node_out_capacities)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_sensed_sender_demands(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([40, 40], dtype=np.float32)
        user2.pos = np.array([0, 0], dtype=np.float32)

        user1_demand = 13e6
        user2_demand = 22e6
        user1.setOfferedLoad(user1_demand)
        user2.setOfferedLoad(user2_demand)
        user1.setDestination(user2)
        user2.setDestination(None)

        # Update and compare
        updateEnv(wrapped_env)
        exp_node_sensed_demands = np.array([0, user1_demand]).reshape(-1, 1)

        res_node_sensed_demands = base_wrapped_env.get_node_sensed_sender_demands(False)

        assert np.array_equal(exp_node_sensed_demands, res_node_sensed_demands)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_sensed_receiver_demands(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([40, 40], dtype=np.float32)
        user2.pos = np.array([0, 0], dtype=np.float32)

        user1_demand = 13e6
        user2_demand = 22e6
        user1.setOfferedLoad(user1_demand)
        user2.setOfferedLoad(user2_demand)
        user1.setDestination(user2)
        user2.setDestination(None)

        # Update and compare
        updateEnv(wrapped_env)
        exp_node_sensed_demands = np.array([user1_demand, 0]).reshape(-1, 1)

        res_node_sensed_demands = base_wrapped_env.get_node_sensed_receiver_demands(
            False
        )

        assert np.array_equal(exp_node_sensed_demands, res_node_sensed_demands)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_user_in_count(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([35, 35], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)

        # Update and compare
        updateEnv(wrapped_env)

        node_user_in_counts = []
        for node in nodes:
            user_in_count = 0
            for user in users:
                user: comps.User
                if user.reaches(node):
                    user_in_count += 1
            node_user_in_counts.append(user_in_count)

        exp_node_user_in_count = np.array(node_user_in_counts).reshape(-1, 1)

        res_node_user_in_count = base_wrapped_env.get_node_user_in_count(False)

        assert np.array_equal(exp_node_user_in_count, res_node_user_in_count)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_user_out_count(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)

        user1: comps.User
        user2: comps.User
        user1.pos = node1.pos + (node1.radius - 1)
        user2.pos = node1.pos + 1

        # Update and compare
        updateEnv(wrapped_env)

        node_user_out_counts = []
        for node in nodes:
            node: comps.MANETNode
            user_out_count = 0
            for user in users:
                if node.reaches(user):
                    user_out_count += 1
            node_user_out_counts.append(user_out_count)

        exp_node_user_out_count = np.array(node_user_out_counts).reshape(-1, 1)

        res_node_user_out_count = base_wrapped_env.get_node_user_out_count(False)

        assert np.array_equal(exp_node_user_out_count, res_node_user_out_count)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_sender_in_count(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([35, 35], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)

        user1.setDestination(None)
        user2.setDestination(user1)

        # Update and compare
        updateEnv(wrapped_env)

        node_sender_in_counts = []
        for node in nodes:
            sender_in_count = 0
            for user in users:
                user: comps.User
                if user.reaches(node) and user.hasDest():
                    sender_in_count += 1
            node_sender_in_counts.append(sender_in_count)

        exp_node_sender_in_count = np.array(node_sender_in_counts).reshape(-1, 1)

        res_node_sender_in_count = base_wrapped_env.get_node_sender_in_count(False)

        assert np.array_equal(exp_node_sender_in_count, res_node_sender_in_count)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_sender_out_count(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)

        user1: comps.User
        user2: comps.User
        user1.pos = node1.pos + (node1.radius - 1)
        user2.pos = node1.pos + 1

        user1.setDestination(None)
        user2.setDestination(user1)

        # Update and compare
        updateEnv(wrapped_env)

        node_sender_out_counts = []
        for node in nodes:
            node: comps.MANETNode
            sender_out_count = 0
            for user in users:
                user: comps.User
                if node.reaches(user) and user.hasDest():
                    sender_out_count += 1
            node_sender_out_counts.append(sender_out_count)

        exp_node_sender_out_count = np.array(node_sender_out_counts).reshape(-1, 1)

        res_node_sender_out_count = base_wrapped_env.get_node_sender_out_count(False)

        assert np.array_equal(exp_node_sender_out_count, res_node_sender_out_count)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_receiver_in_count(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([35, 35], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)
        user1.setDestination(None)
        user2.setDestination(user1)

        dests = {user.dest for user in users if user.hasDest()}

        # Update and compare
        updateEnv(wrapped_env)

        node_receiver_in_counts = []
        for node in nodes:
            receiver_in_count = 0
            for user in users:
                user: comps.User
                if user.reaches(node) and user in dests:
                    receiver_in_count += 1
            node_receiver_in_counts.append(receiver_in_count)

        exp_node_receiver_in_count = np.array(node_receiver_in_counts).reshape(-1, 1)

        res_node_receiver_in_count = base_wrapped_env.get_node_receiver_in_count(False)

        assert np.array_equal(exp_node_receiver_in_count, res_node_receiver_in_count)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_receiver_out_count(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)

        user1: comps.User
        user2: comps.User
        user1.pos = node1.pos + (node1.radius - 1)
        user2.pos = node1.pos + 1

        user1.setDestination(None)
        user2.setDestination(user1)

        dests = {user.dest for user in users if user.hasDest()}

        # Update and compare
        updateEnv(wrapped_env)

        node_receiver_out_counts = []
        for node in nodes:
            node: comps.MANETNode
            receiver_out_count = 0
            for user in users:
                if node.reaches(user) and user in dests:
                    receiver_out_count += 1
            node_receiver_out_counts.append(receiver_out_count)

        exp_node_receiver_out_count = np.array(node_receiver_out_counts).reshape(-1, 1)

        res_node_receiver_out_count = base_wrapped_env.get_node_receiver_out_count(
            False
        )

        assert np.array_equal(exp_node_receiver_out_count, res_node_receiver_out_count)

    def test_get_node_closest_user_dir(self, setUpEnv) -> None:
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        unwrapped_env: envs.Basic_MANETEnv = wrapped_env.unwrapped

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)
        unwrapped_env.node_positions = np.array([node1.pos, node2.pos])

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([40, 35], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)
        unwrapped_env.user_positions = np.array([user1.pos, user2.pos])

        # Update and compare
        updateEnv(wrapped_env)
        node1_closest_dir = np.array(user2.pos) / np.linalg.norm(user2.pos)
        node2_closest_dir = np.array([0, -1], dtype=np.float32)

        exp_node_closest_user_dir = np.array([node1_closest_dir, node2_closest_dir])

        res_node_closest_user_dir = base_wrapped_env.get_node_closest_user_dir(False)

        assert np.allclose(exp_node_closest_user_dir, res_node_closest_user_dir)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_closest_user_position(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        unwrapped_env: envs.Basic_MANETEnv = wrapped_env.unwrapped

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)
        unwrapped_env.node_positions = np.array([node1.pos, node2.pos])

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([40, 35], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)
        unwrapped_env.user_positions = np.array([user1.pos, user2.pos])

        # Update and compare
        updateEnv(wrapped_env)

        exp_node_closest_user_position = np.array([user2.pos, user1.pos])

        res_node_closest_user_position = (
            base_wrapped_env.get_node_closest_user_position(False)
        )

        assert np.allclose(
            exp_node_closest_user_position, res_node_closest_user_position
        )

    def test_get_node_closest_sender_dir(self, setUpEnv) -> None:
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        unwrapped_env: envs.Basic_MANETEnv = wrapped_env.unwrapped

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)
        unwrapped_env.node_positions = np.array([node1.pos, node2.pos])

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([40, 39.5], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)
        user1.setDestination(user2)
        user2.setDestination(None)
        unwrapped_env.user_positions = np.array([user1.pos, user2.pos])

        # Update and compare
        updateEnv(wrapped_env)
        dir_node1_to_user1 = user1.pos - node1.pos
        node1_closest_sender_dir = dir_node1_to_user1 / np.linalg.norm(
            dir_node1_to_user1
        )
        node2_closest_sender_dir = np.array([0, -0.5], dtype=np.float32)

        exp_node_closest_sender_dir = np.array(
            [node1_closest_sender_dir, node2_closest_sender_dir]
        )

        res_node_closest_sender_dir = base_wrapped_env.get_node_closest_sender_dir(
            False
        )

        assert np.allclose(exp_node_closest_sender_dir, res_node_closest_sender_dir)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_closest_sender_position(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        unwrapped_env: envs.Basic_MANETEnv = wrapped_env.unwrapped

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)
        unwrapped_env.node_positions = np.array([node1.pos, node2.pos])

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([40, 35], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)
        user1.setDestination(user2)
        user2.setDestination(None)
        unwrapped_env.user_positions = np.array([user1.pos, user2.pos])

        # Update and compare
        updateEnv(wrapped_env)

        exp_node_closest_sender_position = np.array([user1.pos, user1.pos])

        res_node_closest_sender_position = (
            base_wrapped_env.get_node_closest_sender_position(False)
        )

        assert np.allclose(
            exp_node_closest_sender_position, res_node_closest_sender_position
        )

    def test_get_node_closest_receiver_dir(self, setUpEnv) -> None:
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        unwrapped_env: envs.Basic_MANETEnv = wrapped_env.unwrapped

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)
        unwrapped_env.node_positions = np.array([node1.pos, node2.pos])

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([40, 39.5], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)
        user1.setDestination(None)
        user2.setDestination(user1)
        unwrapped_env.user_positions = np.array([user1.pos, user2.pos])

        # Update and compare
        updateEnv(wrapped_env)
        dir_node1_to_user1 = user1.pos - node1.pos
        node1_closest_receiver_dir = dir_node1_to_user1 / np.linalg.norm(
            dir_node1_to_user1
        )
        node2_closest_receiver_dir = np.array([0, -0.5], dtype=np.float32)

        exp_node_closest_receiver_dir = np.array(
            [node1_closest_receiver_dir, node2_closest_receiver_dir]
        )

        res_node_closest_receiver_dir = base_wrapped_env.get_node_closest_receiver_dir(
            False
        )

        assert np.allclose(exp_node_closest_receiver_dir, res_node_closest_receiver_dir)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_closest_receiver_position(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        unwrapped_env: envs.Basic_MANETEnv = wrapped_env.unwrapped

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)
        unwrapped_env.node_positions = np.array([node1.pos, node2.pos])

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([40, 35], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)
        user1.setDestination(None)
        user2.setDestination(user1)
        unwrapped_env.user_positions = np.array([user1.pos, user2.pos])

        # Update and compare
        updateEnv(wrapped_env)

        exp_node_closest_receiver_position = np.array([user1.pos, user1.pos])

        res_node_closest_receiver_position = (
            base_wrapped_env.get_node_closest_receiver_position(False)
        )

        assert np.allclose(
            exp_node_closest_receiver_position, res_node_closest_receiver_position
        )

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_node_closest_user_interference(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        unwrapped_env: envs.Basic_MANETEnv = wrapped_env.unwrapped

        # Set node and user positions
        node1, node2 = nodes
        user1, user2 = users

        node1: comps.MANETNode
        node2: comps.MANETNode
        node1.pos = np.array([0, 0], dtype=np.float32)
        node2.pos = np.array([40, 40], dtype=np.float32)
        unwrapped_env.node_positions = np.array([node1.pos, node2.pos])

        user1: comps.User
        user2: comps.User
        user1.pos = np.array([40, 35], dtype=np.float32)
        user2.pos = np.array([40, 30], dtype=np.float32)
        unwrapped_env.user_positions = np.array([user1.pos, user2.pos])

        # Update and compare
        updateEnv(wrapped_env)

        exp_node_closest_user_interference = np.array(
            [user2.interferenceReceived_mW, user1.interferenceReceived_mW]
        ).reshape(-1, 1)

        res_node_closest_user_interference = (
            base_wrapped_env.get_node_closest_user_interference(False)
        )

        assert np.allclose(
            exp_node_closest_user_interference, res_node_closest_user_interference
        )

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_user_position(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_user_positions = np.array([user.pos for user in users])

        res_user_positions = base_wrapped_env.get_user_position(False)

        assert np.array_equal(exp_user_positions, res_user_positions)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_user_signal_received(
        self, mock_norm_transform: Mock, setUpEnv
    ) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_user_signals = np.array([user.signalReceived_mW for user in users]).reshape(
            -1, 1
        )

        res_user_signals = base_wrapped_env.get_user_signal_received(False)

        assert np.array_equal(exp_user_signals, res_user_signals)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_user_interference(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_user_interferences = np.array(
            [user.interferenceReceived_mW for user in users]
        ).reshape(-1, 1)

        res_user_interferences = base_wrapped_env.get_user_interference(False)

        assert np.array_equal(exp_user_interferences, res_user_interferences)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_user_dest_indices(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        user1, user2 = users

        user1: comps.User
        user2: comps.User

        user1.setDestination(user2)
        user2.dest = None

        # Update and compare
        updateEnv(wrapped_env)
        exp_user_dest_indices = np.array([1, 2]).reshape(-1, 1)

        res_user_dest_indices = base_wrapped_env.get_user_dest_indices(False)

        assert np.array_equal(exp_user_dest_indices, res_user_dest_indices)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_user_dest_position(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        user1, user2 = users

        user1: comps.User
        user2: comps.User

        user1.setDestination(user2)
        user2.dest = None

        # Update and compare
        updateEnv(wrapped_env)
        exp_user_dest_position = np.array([user2.pos, np.zeros((2,))])

        res_user_dest_position = base_wrapped_env.get_user_dest_position(False)

        assert np.array_equal(exp_user_dest_position, res_user_dest_position)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_user_signal_power(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_user_signalPow = np.array([user.signalPower_mW for user in users]).reshape(
            -1, 1
        )

        res_user_signalPow = base_wrapped_env.get_user_signal_power(False)

        assert np.array_equal(exp_user_signalPow, res_user_signalPow)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_user_offered_load(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        updateEnv(wrapped_env)
        exp_user_offered_load = np.array([user.offeredLoad for user in users]).reshape(
            -1, 1
        )

        res_user_offered_load = base_wrapped_env.get_user_offered_load(False)

        assert np.array_equal(exp_user_offered_load, res_user_offered_load)

    @patch("MANET.ObservationWrappers.normalize_and_transform")
    def test_get_capacity_matrix(self, mock_norm_transform: Mock, setUpEnv) -> None:
        mock_norm_transform.side_effect = self.mock_normalize_and_transform
        wrapped_env, network, nodes, users = setUpEnv
        wrapped_env: ow.Configurable_ObservationWrapper
        base_wrapped_env: ow.Base_ObservationWrapper = wrapped_env.wrapped_env

        # Update and compare
        updateEnv(wrapped_env)

        network: net.Network
        graph = network.getGraph()
        exp_capacity_matrix = np.zeros((4, 4))

        for i, comp_i in enumerate(np.concatenate([nodes, users])):
            comp_i: comps.NetworkComponent
            for j, comp_j in enumerate(np.concatenate([nodes, users])):
                comp_j: comps.NetworkComponent
                exp_capacity_matrix[i, j] = (
                    graph.get_edge_data(comp_i, comp_j)["capacity"]
                    if graph.has_edge(comp_i, comp_j)
                    else 0
                )

        res_exp_capacity_matrix = base_wrapped_env.get_capacity_matrix(False)

        assert np.allclose(exp_capacity_matrix, res_exp_capacity_matrix)
