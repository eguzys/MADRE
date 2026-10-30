# Need for testing
from unittest.mock import patch, MagicMock
import pytest

# Standard libraries
import numpy as np

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


def create_non_wrapped_env(
    numbOfNodes=5, numbOfJammers=2, numbOfUsers=10
) -> envs.Basic_MANETEnv:
    """Helper function to create environment quickly for tests

    Args:
        numbOfNodes (int, optional): The number of nodes in the environment. Defaults to 5.
        numbOfJammers (int, optional): The number of nodes in the environment. Defaults to 2.
        numbOfUsers (int, optional): The number of nodes in the environment. Defaults to 10.

    Returns:
        envs.Basic_MANETEnv: The create MANET environment
    """
    env = envs.Basic_MANETEnv(
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        render_mode=None,
        seed=42,
    )
    env.reset(seed=42)
    return env


def create_wrapped_env(numbOfNodes=5, numbOfJammers=2, numbOfUsers=10) -> gym.Env:
    """Helper method to easily create a wrapped environment

    Args:
        numbOfNodes (int, optional): The number of nodes in the environment. Defaults to 5.
        numbOfJammers (int, optional): The number of nodes in the environment. Defaults to 2.
        numbOfUsers (int, optional): The number of nodes in the environment. Defaults to 10.

    Returns:
        gym.Env: The wrapped MANET Environment
    """
    env = envs.Basic_MANETEnv(
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        render_mode=None,
        seed=42,
    )
    env.reset(seed=42)
    env = ut.wrapEnv(
        env=env, rew_wrapper=rw.MovementCost, obs_wrapper=ow.VS_COW, use_monitor=False
    )
    return env


@pytest.fixture
def wrapped_env() -> gym.Env:
    """Pytest fixture to create a wrapped environment with 5 nodes, 2 jammers and 10 users.

    Returns:
        gym.Env: The wrapped MANET Environment
    """
    env = create_wrapped_env()

    return env


@pytest.fixture
def non_wrapped_env() -> envs.Basic_MANETEnv:
    """Pytest fixture to create a non-wrapped environment with 5 nodes, 2 jammers and 10 users.

    Returns:
        envs.Basic_MANETEnv: The MANET environment
    """
    env = create_non_wrapped_env()

    return env


@pytest.fixture
def non_wrapped_uninit_env() -> envs.Basic_MANETEnv:
    """Fixture to create a non-wrapped not yet initialized MANET environment with 5 nodes, 2 jammers and 10 users.
    This is done by mocking the reset function, which basically initiates all environment properties.

    Returns:
        envs.Basic_MANETEnv: The MANET environment
    """
    with patch("MANET.Environments.Basic_MANETEnv.reset") as reset_mock:
        reset_mock.return_value = (
            None,
            None,
        )  # Mock reset to prevent full initialization

        # Initialize the environment but without `reset` running fully
        env = (
            create_non_wrapped_env()
        )  # Or use `create_non_wrapped_env` if you want an unwrapped environment

        return env


@pytest.fixture
def wrapped_uninit_env() -> gym.Env:
    """Fixture to create a non-wrapped not yet initialized MANET environment with 5 nodes, 2 jammers and 10 users.
    This is done by mocking the reset function, which basically initiates all environment properties.

    Returns:
        gym.Env: The wrapped MANET environment
    """
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


class Test_Init:
    def test_initialization_without_reset(self, non_wrapped_uninit_env: gym.Env):
        """Test that all attributes exist but are not yet initialized.

        Args:
            non_wrapped_uninit_env (gym.Env): creates a wrapped MANET environment
        """
        env: envs.Basic_MANETEnv = non_wrapped_uninit_env

        # RL properties should be initialized
        assert env.action_space is not None
        assert env.observation_space is None

        # Episodes properties should not yet be initialized
        assert env.ep_length == np.inf
        assert env.ep_step is None

        # Check that actor properties are not yet initialized
        assert env.node_positions is None
        assert env.nodes is None

        assert env.jammer_positions is None
        assert env.jammers is None

        assert env.users is None
        assert env.user_positions is None
        assert env.user_to_idx is None

        assert env.clusterSizes is None

        # The graph should be created
        assert env.network is None

    def test_initialization_with_reset(self):
        """
        Given that the reset method is executed correctly all variables should not be None except for the ones that are specifically set to None
        """
        numbOfNodes = 5
        numbOfJammers = 2
        numbOfUsers = 10
        render_mode = None
        seed = 42
        unwrapped_env = envs.Basic_MANETEnv(
            numbOfNodes=numbOfNodes,
            numbOfJammers=numbOfJammers,
            numbOfUsers=numbOfUsers,
            render_mode=render_mode,
            seed=seed,
        )

        # Number of components should be set correctly
        assert unwrapped_env.numbOfNodes == numbOfNodes
        assert unwrapped_env.numbOfJammers == numbOfJammers
        assert unwrapped_env.numbOfUsers == numbOfUsers

        # Render mode
        assert unwrapped_env.render_mode == render_mode

        # Action space should be initialized but observation space shouldn't
        assert unwrapped_env.action_space is not None
        assert unwrapped_env.observation_space is None

        # Episodes properties should be initialized
        assert unwrapped_env.ep_length == np.inf
        assert unwrapped_env.ep_step == 0

        # Check that actor properties are initialized
        assert unwrapped_env.node_positions.shape == (unwrapped_env.numbOfNodes, 2)
        assert isinstance(unwrapped_env.node_positions, np.ndarray)

        assert unwrapped_env.nodes.shape == (unwrapped_env.numbOfNodes,)
        assert isinstance(unwrapped_env.nodes, np.ndarray)
        assert np.all(isinstance(node, comps.MANETNode) for node in unwrapped_env.nodes)

        assert np.array_equal(
            unwrapped_env.lastAction, np.zeros((unwrapped_env.numbOfNodes, 2))
        )

        assert unwrapped_env.jammer_positions.shape == (unwrapped_env.numbOfJammers, 2)
        assert isinstance(unwrapped_env.node_positions, np.ndarray)

        assert unwrapped_env.jammers.shape == (unwrapped_env.numbOfJammers,)
        assert isinstance(unwrapped_env.jammers, np.ndarray)
        assert np.all(
            isinstance(jammer, comps.Jammer) for jammer in unwrapped_env.jammers
        )

        assert unwrapped_env.user_positions.shape == (unwrapped_env.numbOfUsers, 2)
        assert isinstance(unwrapped_env.user_positions, np.ndarray)

        assert unwrapped_env.users.shape == (unwrapped_env.numbOfUsers,)
        assert isinstance(unwrapped_env.users, np.ndarray)
        assert np.all(isinstance(user, comps.User) for user in unwrapped_env.users)

        assert unwrapped_env.user_to_idx is not None

        assert isinstance(unwrapped_env.clusterSizes, np.ndarray)

        # The graph should be created
        assert isinstance(unwrapped_env.network, net.Network)


class Test_Step:
    def test_return_types(self, wrapped_env: gym.Env):
        """Check that the return types are correct for all values returned by the step function.

        Args:
            wrapped_env (gym.Env): The wrapped MANET environment
        """

        env: gym.Env = wrapped_env
        unwrappped_env: envs.Basic_MANETEnv = env.unwrapped

        action = np.zeros((unwrappped_env.numbOfNodes, 2))
        observation, reward, terminated, truncated, info = env.step(action)

        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observations_change_after_step(self, wrapped_env: gym.Env):
        """
        Check that the observation changes after a step.

        Args:
            wrapped_env (gym.Env): The wrapped MANET environment
        """
        env: gym.Env = wrapped_env
        unwrappped_env: envs.Basic_MANETEnv = env.unwrapped

        env.reset(seed=42)
        first_observation = env.get_wrapper_attr("observation")()
        action = np.ones((unwrappped_env.numbOfNodes, 2))
        second_observation, _, _, _, _ = env.step(action)

        assert not np.array_equal(
            first_observation, second_observation
        ), "Observation should have changed."


class Test_Reset:
    def test_params_were_reset(self, non_wrapped_env: envs.Basic_MANETEnv):
        """Ensure that the reset function correctly resets certain parameters.

        Args:
            non_wrapped_env (envs.Basic_MANETEnv): The MANET environment.
        """
        env: envs.Basic_MANETEnv = non_wrapped_env

        nodes = env.nodes
        jammers = env.jammers
        users = env.users

        env.reset(seed=42)

        assert env.ep_step == 0

        assert env.nodes is not nodes
        assert env.jammers is not jammers
        assert env.users is not users

    def test_reset_returns_correct_params(self, non_wrapped_env: envs.Basic_MANETEnv):
        """Ensure the parameters returned by the reset function are of correct type.

        Args:
            non_wrapped_env (envs.Basic_MANETEnv): The MANET environment.
        """
        env: envs.Basic_MANETEnv = non_wrapped_env
        env = ut.wrapEnv(env=env, rew_wrapper=rw.SquaredReward, obs_wrapper=ow.VS_COW)
        observation, info = env.reset(seed=42)

        assert isinstance(observation, np.ndarray)
        assert isinstance(info, dict)


class Test_Done:
    def test_done(self, non_wrapped_env: envs.Basic_MANETEnv):
        """Ensure that the done flag is set correctly. Should only by set to true if the maximum throughput is achieved
        in the Static_MANETEnv. Hence, it should always be False.

        Args:
            non_wrapped_env (envs.Basic_MANETEnv): The MANET environment.
        """
        env: envs.Basic_MANETEnv = non_wrapped_env

        ep_length = 100
        env.set_ep_length(ep_length)
        action = env.action_space.sample()
        env.step(action)

        env.setEpStep(ep_length)
        assert env.done() is False

        env.setEpStep(0)
        assert env.done() is False


class Test_Truncated:
    def test_truncated(self, non_wrapped_env: envs.Basic_MANETEnv):
        """
        Ensure that the truncated flag is set, when the ep_length is reached.

        Args:
            non_wrapped_env (envs.Basic_MANETEnv): The MANET environment.
        """
        env: envs.Basic_MANETEnv = non_wrapped_env

        ep_length = 100
        env.set_ep_length(ep_length)

        # Should be true if the episode step is equal to the episode length
        env.setEpStep(ep_length)
        assert env.truncated() is True

        env.setEpStep(0)
        assert env.truncated() is False


class Test_Info:
    def test_getInfo(self, non_wrapped_env: envs.Basic_MANETEnv):
        """Ensure that the getInfo method returns an empty string.

        Args:
            non_wrapped_env (envs.Basic_MANETEnv): The MANET environment.
        """
        env: envs.Basic_MANETEnv = non_wrapped_env
        info = env.info()
        assert isinstance(info, dict)
        assert info == {}, "info should be an empty dict"


class Test_Render:
    # TODO: not priority
    pass


class Test_SetEpLength:
    def test_set_ep_length(self, non_wrapped_env: envs.Basic_MANETEnv):
        """Ensure that ep_length is set correctly with its setter

        Args:
            non_wrapped_env (envs.Basic_MANETEnv): The MANET environment.
        """
        env: envs.Basic_MANETEnv = non_wrapped_env

        assert env.ep_length == np.inf

        env.set_ep_length(100)
        assert env.ep_length == 100


class Test_InitializeComponents:
    def test_initializeComponents(self, non_wrapped_uninit_env: envs.Basic_MANETEnv):
        """
        Check that components are initialized and have the expected shape and are of expected instance.

        Args:
            non_wrapped_uninit_env (envs.Basic_MANETEnv): The MANET environment without the attributes initialized yet.
        """
        self.env: envs.Basic_MANETEnv = non_wrapped_uninit_env

        self.check_comps_not_initialized()

        self.env.initializeComponents()

        self.check_comps_initialized()

    def check_comps_not_initialized(self):
        """Helper method that asserts that the components and their positions are None."""
        assert self.env.node_positions is None
        assert self.env.jammer_positions is None
        assert self.env.user_positions is None

        assert self.env.nodes is None
        assert self.env.jammers is None
        assert self.env.users is None

    def check_comps_initialized(self):
        """Helper method that asserts that all components (nodes, jammers and users) are correctly initialized in that
        their arrays has the correct shape and store the correct type of NetworkComponent.
        """
        assert self.env.node_positions.shape == (self.env.numbOfNodes, 2)
        assert isinstance(self.env.node_positions, np.ndarray)

        assert self.env.nodes.shape == (self.env.numbOfNodes,)
        assert isinstance(self.env.nodes, np.ndarray)
        assert np.all(isinstance(node, comps.MANETNode) for node in self.env.nodes)

        assert self.env.jammer_positions.shape == (self.env.numbOfJammers, 2)
        assert isinstance(self.env.node_positions, np.ndarray)

        assert self.env.jammers.shape == (self.env.numbOfJammers,)
        assert isinstance(self.env.jammers, np.ndarray)
        assert np.all(isinstance(jammer, comps.Jammer) for jammer in self.env.jammers)

        assert self.env.user_positions.shape == (self.env.numbOfUsers, 2)
        assert isinstance(self.env.user_positions, np.ndarray)

        assert self.env.users.shape == (self.env.numbOfUsers,)
        assert isinstance(self.env.users, np.ndarray)
        assert np.all(isinstance(user, comps.User) for user in self.env.users)


class Test_CreateComponents:
    # Not worth testing
    pass


class Test_DistributeComponents:
    # Not worth testing
    pass


class Test_DistributeNodes:
    def test_check_positions_set_correctly(
        self, non_wrapped_uninit_env: envs.Basic_MANETEnv
    ):
        """Check that distributing the nodes results in the nodes' positions being set.

        Args:
            non_wrapped_uninit_env (envs.Basic_MANETEnv): The MANET environment without the attributes initialized yet.
        """
        env: envs.Basic_MANETEnv = non_wrapped_uninit_env

        assert env.node_positions is None
        assert env.nodes is None

        env.distributeNodes()

        assert env.node_positions.shape == (env.numbOfNodes, 2)
        assert isinstance(env.node_positions, np.ndarray)

        assert env.nodes is None


class Test_DistributeUsers:
    def test_attr_set_correctly(self, non_wrapped_uninit_env: envs.Basic_MANETEnv):
        """Check that distributing the users results in the users' positions being set.

        Args:
            non_wrapped_uninit_env (envs.Basic_MANETEnv): The MANET environment without the attributes initialized yet.
        """
        env: envs.Basic_MANETEnv = non_wrapped_uninit_env

        assert env.clusterSizes is None
        assert env.user_positions is None
        assert env.users is None

        env.distributeUsers()

        assert env.user_positions.shape == (env.numbOfUsers, 2)
        assert isinstance(env.user_positions, np.ndarray)
        assert env.clusterSizes is not None
        assert isinstance(env.clusterSizes, np.ndarray)

        assert env.users is None


class Test_DetermineNumbOfClusters:
    @pytest.mark.parametrize("alpha", [0, 0.1, 0.2, 0.5, 0.8, 1])
    def test_boundaries(self, alpha, non_wrapped_env, numb_of_tests=15):
        """Check that for different alpha values the numb of clusters
        is within the defined bounderies.

        Args:
            alpha (float): determines the distribution of the clusters.
            non_wrapped_env (envs.Basic_MANETEnv): The MANET environment.
            numb_of_tests (int, optional): How often to check for a specific alpha value. Defaults to 15.
        """
        env: envs.Basic_MANETEnv = non_wrapped_env
        for _ in range(numb_of_tests):
            # Call the method to test
            result = env.determineNumbOfClusters(alpha=alpha)

            min_clusters = 2
            max_clusters = env.numbOfUsers
            assert result >= min_clusters
            assert result <= max_clusters

    def test_returnType(self, non_wrapped_env: envs.Basic_MANETEnv):
        """Check that it always returns an interger value.

        Args:
            non_wrapped_env (envs.Basic_MANETEnv): The MANET environment.
        """
        env: envs.Basic_MANETEnv = non_wrapped_env
        result = env.determineNumbOfClusters()
        assert isinstance(result, int)


class Test_RandChooseClusterSizes:
    @pytest.mark.parametrize(
        "numbOfClusters, numbOfUsers", [(5, 10), (10, 15), (3, 5), (30, 100)]
    )
    def test_sum_of_cluster_sizes(
        self, numbOfClusters, numbOfUsers, non_wrapped_uninit_env
    ):
        """Ensure that the cluster sizes always sum up to the number of users.

        Args:
            numbOfClusters (int): Number of clusters to distribute the users over.
            numbOfUsers (int): Number of users to distribute over all clusters
            non_wrapped_uninit_env (envs.Basic_MANETEnv): The MANET environment without the attributes initialized yet.
        """
        env: envs.Basic_MANETEnv = non_wrapped_uninit_env
        env.setNumbOfUsers(numbOfUsers)

        numbOfSteps = 20
        for _ in range(numbOfSteps):
            clusterSizes = env.randChooseClusterSizes(numbOfClusters, beta=0.1)

            assert sum(clusterSizes) == numbOfUsers


class Test_RandDistrClusterCenters:

    @pytest.mark.parametrize("numbOfClusters", [1, 2, 3, 5, 8, 12])
    def test_check_centers_set_correctly(self, numbOfClusters, non_wrapped_uninit_env):
        """
        Test that the cluster centers are set correctly:
        - The shape of the cluster centers array matches the number of clusters.
        - The cluster centers are of type numpy.ndarray.
        - Each cluster center lies within the bounds [0, env.SIZE]^2.

        Args:
            numbOfClusters (int): Number of clusters to distribute.
            non_wrapped_uninit_env (envs.Basic_MANETEnv): The MANET environment without the attributes initialized yet.
        """
        env: envs.Basic_MANETEnv = non_wrapped_uninit_env

        clusterCenters = env.randDistrClusterCenters(numbOfClusters)

        # Check the shape of the cluster centers array
        assert clusterCenters.shape == (numbOfClusters, 2)

        # Check the type of the cluster centers array
        assert isinstance(clusterCenters, np.ndarray)

        # Check that each cluster center is within the bounds [0, env.SIZE]^2
        assert np.all((clusterCenters >= 0) & (clusterCenters <= env.SIZE))


class Test_DistributeUsersOverClusters:
    @pytest.mark.parametrize(
        "numbOfNodes, numbOfJammers, numbOfUsers",
        [(2, 3, 4), (2, 1, 8), (3, 1, 3), (20, 2, 100)],
    )
    def test_check_positions_set_correctly(
        self, numbOfNodes, numbOfJammers, numbOfUsers, non_wrapped_uninit_env
    ):
        """Check that the users a distributed such that their position is set but their object
        isn't yet initialized.

        Args:
            numbOfNodes (int): The number of nodes to create
            numbOfJammers (int): The number of jammers to create
            numbOfUsers (int): The number of users to create
            non_wrapped_uninit_env (envs.Basic_MANETEnv): The MANET environment without the attributes initialized yet.
        """
        env: envs.Basic_MANETEnv = non_wrapped_uninit_env

        env.setNumbOfNodes(numbOfNodes)
        env.setNumbOfJammers(numbOfJammers)
        env.setNumbOfUsers(numbOfUsers)

        numbOfClusters = env.determineNumbOfClusters()
        clusterSizes = env.randChooseClusterSizes(numbOfClusters)
        env.setClusterSizes(clusterSizes)
        clusterCenters = env.randDistrClusterCenters(numbOfClusters)

        assert env.user_positions is None
        assert env.users is None

        user_positions: np.ndarray = env.distributeUsersOverClusters(clusterCenters)

        assert user_positions.shape == (env.numbOfUsers, 2)
        assert isinstance(user_positions, np.ndarray)
        # Check that each user is within the bounds [0, env.SIZE]^2
        assert np.all((user_positions >= 0) & (user_positions <= env.SIZE))

        assert env.users is None


class Test_DistributeJammers:
    @pytest.mark.parametrize("numbOfJammers", [1, 2, 4, 7, 13])
    def test_less_jammers_than_users(self, numbOfJammers):
        """Test the jammers distribution, for when their are less jammers than users.

        Args:
            numbOfJammers (int): The number of jammers to distribute.
        """
        np.random.seed(42)
        numbOfUsers = numbOfJammers + np.random.randint(0, 101)

        env = create_non_wrapped_env(
            numbOfNodes=2, numbOfJammers=numbOfJammers, numbOfUsers=numbOfUsers
        )

        env.distributeJammers()

        jammer_positions = env.jammer_positions

        assert jammer_positions.shape == (env.numbOfJammers, 2)
        assert isinstance(jammer_positions, np.ndarray)
        assert np.all((jammer_positions >= 0) & (jammer_positions <= env.SIZE))

    @pytest.mark.parametrize("numbOfUsers", [2, 4, 7, 13])
    def test_more_jammers_than_users(self, numbOfUsers):
        """Check that jammers positions are initialized correctly with more jammers than users.

        Args:
            numbOfUsers (int): The number of users in the environment
        """
        np.random.seed(42)
        numbOfJammers = numbOfUsers + np.random.randint(0, 20)

        env = create_non_wrapped_env(
            numbOfNodes=2, numbOfJammers=numbOfJammers, numbOfUsers=numbOfUsers
        )

        env.distributeJammers()

        jammer_positions = env.jammer_positions

        assert jammer_positions.shape == (env.numbOfJammers, 2)
        assert isinstance(jammer_positions, np.ndarray)
        assert np.all((jammer_positions >= 0) & (jammer_positions <= env.SIZE))


class Test_MoveNodes:
    def test_moveNodes(self, non_wrapped_env: envs.Basic_MANETEnv):
        """Test that the nodes are moved according to the action specified.

        Args:
            non_wrapped_env (envs.Basic_MANETEnv): The MANET environment.
        """
        env: envs.Basic_MANETEnv = non_wrapped_env

        numbOfSteps = 20

        for _ in range(numbOfSteps):
            node_positions = env.node_positions.copy()
            action = env.action_space.sample()

            env.moveNodes(action)

            new_node_positions = env.node_positions

            assert not np.array_equal(new_node_positions, node_positions)


class Test_MoveUsers:
    # Not implemented in this class
    pass


class Test_MoveJammers:
    # Not implemented in this class
    pass


class Test_InitializeDemand:
    def test_check_values(self, non_wrapped_uninit_env: envs.Basic_MANETEnv):
        """Check that the constand demand is correctly initialized.

        Args:
            non_wrapped_uninit_env (envs.Basic_MANETEnv): The MANET environment without the attributes initialized yet.
        """
        env: envs.Basic_MANETEnv = non_wrapped_uninit_env

        env.initializeComponents()

        users: np.ndarray = env.users

        env.initializeDemand()

        offeredLoads = [user.offeredLoad for user in users]

        assert np.all([offeredLoad == 20 * 1e6 for offeredLoad in offeredLoads])


class Test_initializeSendersAndReceivers:
    def test_user_dest_not_oneself(self, non_wrapped_uninit_env: envs.Basic_MANETEnv):
        """Test that the destinations are intialized and such that a user never sends to itself.

        Args:
            non_wrapped_uninit_env (envs.Basic_MANETEnv): The MANET environment without the attributes initialized yet.
        """
        num_of_steps = 15

        for _ in range(num_of_steps):
            env: envs.Basic_MANETEnv = non_wrapped_uninit_env

            env.initializeComponents()

            users = env.users

            env.initializeDemand()
            env.initializeSendersAndReceivers()

            destinations = [user.dest for user in users]

            for user, dest in zip(users, destinations):
                assert user != dest


class Test_InitializeNormFactors:
    # Not worth testing
    pass


class Test_StartGui:
    # Not priority
    pass


class Test_StopGuiThread:
    # Not priority
    pass


class Test_Del:
    # Not priority
    pass


class Test_GetRewWrapperName:
    # Not worth testing
    pass


class Test_GetObsWrapperName:
    # Not worth testing
    pass
