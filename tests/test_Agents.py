# Standard libraries
import numpy as np

# Gymnasium
import gymnasium as gym

# Own classes
import MANET.Components as comps
import MANET.Network as net
import MANET.Environments as envs
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw
import MANET.Attackers as at
import MANET.Agents as ag
import MANET.ConditionStrategy as cs

# Utils
import Utils.training_utils as ut


class Test_OutAndThroughAgent:

    def test_nodeSettleAfterMovingAway(self):
        """Tests whether the node correctly settles after moving and of the jammers radius, since there are two users, that would provide the best position.
        So, this would be case 1 in which the node finds a viable position after first moving out of the jammers radius.
        """
        # Set positions
        node_positions = np.array([[50, 50]]).astype(np.float32)
        jammer_positions = np.array([[40, 50]]).astype(np.float32)
        user_positions = np.array([[65, 50], [65, 50]]).astype(np.float32)

        # Initialize env and agent
        env, agent = create_env_and_agent(
            node_positions,
            jammer_positions,
            user_positions,
            agent_type="OutAndThroughAgent",
        )
        agent: ag.AgentBase
        agent.setConditionStrat(cs.SINRConditionStrategy(threshold_sinr=5))
        env: gym.Env
        unwrapped_env: envs.Basic_MANETEnv = env.unwrapped

        assert (
            agent is not None
        ), "agent isn't correctly initialized, agent object is None"
        observedNode: comps.MANETNode = unwrapped_env.nodes[0]

        # Check whether sampling is done correctly
        relative_positions = np.array(
            [
                [0, 1],
                [-0.5, 0.5],
                [-1, 0],
                [-0.5, -0.5],
                [0, -1],
                [0.5, -0.5],
                [1, 0],
                [0.5, 0.5],
                [0, 0],
            ]
        ).astype(np.float32)
        first_eigth_positions = relative_positions + observedNode.pos

        checkPositionsForNode(first_eigth_positions, observedNode, env, agent)

        # Check whether sampling resulted in correct directions and moves are correct
        doEnvironmentSteps(env, agent, agent.numb_of_sampling_steps)
        assert np.array_equal(
            observedNode.pos,
            np.array([50 + agent.numb_of_sampling_steps, 50], dtype=np.float32),
        ), "node has not ended up correctly"

        # Check that node takes position closest to the two users
        closest_pos = (
            np.array([65, 50], dtype=np.float32)
            if agent.numb_of_sampling_steps >= 20
            else np.array([50 + agent.numb_of_sampling_steps, 50], dtype=np.float32)
        )
        doEnvironmentSteps(env, agent, 5)
        assert np.array_equal(
            observedNode.pos, closest_pos
        ), "node has not moved into the optimal position"

    def test_nodeSettleAfterMovingThrough(self):
        """Test case 2 which is when the node doesn't find a viable option after first moving away from the jammers position and then goes through on the other
        side where it finds a viable position."""
        # Initialize positions
        node_positions = np.array([[50, 50]]).astype(np.float32)
        jammer_positions = np.array([[48, 50]]).astype(np.float32)
        user_positions = np.array([[40, 50], [40, 50]]).astype(np.float32)

        # Create environment and agent
        env, agent = create_env_and_agent(
            node_positions,
            jammer_positions,
            user_positions,
            agent_type="OutAndThroughAgent",
        )
        env: gym.Env
        unwrapped_env: envs.Basic_MANETEnv = env.unwrapped

        assert (
            agent is not None
        ), "agent isn't correctly initialized, agent object is None"

        # Check sampling works correctly
        observedNode: comps.MANETNode = unwrapped_env.nodes[0]

        relative_positions = np.array(
            [
                [0, 1],
                [-0.5, 0.5],
                [-1, 0],
                [-0.5, -0.5],
                [0, -1],
                [0.5, -0.5],
                [1, 0],
                [0.5, 0.5],
                [0, 0],
            ]
        ).astype(np.float32)
        first_eigth_positions = relative_positions + observedNode.pos

        checkPositionsForNode(first_eigth_positions, observedNode, env, agent)

        # Check node moves in the correct direction
        exp_dir = np.array([1, 0], dtype=np.float32)
        exp_pos = observedNode.pos + exp_dir * agent.numb_of_sampling_steps
        doEnvironmentSteps(env, agent, agent.numb_of_sampling_steps)
        assert np.array_equal(
            observedNode.pos, exp_pos
        ), "node has not ended up at expected position"

        # Check node goes through
        exp_pos = observedNode.pos - exp_dir * 2 * agent.numb_of_sampling_steps
        doEnvironmentSteps(env, agent, 2 * agent.numb_of_sampling_steps)
        assert np.array_equal(
            observedNode.pos, exp_pos
        ), "node has not ended up at expected position"

        # Check node goes to users, where signal is the strongest.
        doEnvironmentSteps(env, agent, agent.numb_of_sampling_steps)
        assert np.array_equal(
            observedNode.pos, user_positions[0]
        ), "node should not have moved away"


class Test_RandomSampler:
    def test_moveAwayFromNoise_withoutSignal(self):
        """
        Check that the node, with the random sampler, chooses the position among the sampled positions
        that is furthest away from the jammers, as it is maximizing for SINR.
        """
        # Create mock positions
        mock_positions = np.array([[30, 50], [40, 50], [60, 50]])

        # Initialize postions
        node_positions = np.array([[50, 50]]).astype(np.float32)
        jammer_positions = np.array([[40, 50]]).astype(np.float32)
        user_positions = np.array([[0, 0], [0, 0]]).astype(np.float32)

        # Create env and agent
        env, agent = create_env_and_agent(
            node_positions, jammer_positions, user_positions, agent_type="RandomSampler"
        )
        env: gym.Env
        unwrapped_env: envs.Basic_MANETEnv = env.unwrapped
        agent: ag.AgentBase

        # Mock the agent's random generator
        class MockRandomGen:
            def multivariate_normal(self, mean, cov, size):
                return mock_positions

        agent.np_random_gen = MockRandomGen()

        assert (
            agent is not None
        ), "agent isn't correctly initialized, agent object is None"

        # Check that it moves to position furthest away from jammer
        observedNode: comps.MANETNode = unwrapped_env.nodes[0]

        doEnvironmentSteps(env, agent, 50)
        assert np.array_equal(
            observedNode.pos, np.array([60, 50], dtype=np.float32)
        ), "not has not chosen the expected optimal position"

    def test_findCorrectPositions(self):
        """
        Check that the RandomSampler Agent finds the best postion, which is the one closest to the users.
        """
        # Set up mocked positions
        mock_positions = np.array([[30, 50], [40, 50], [60, 50]])  # Mocked output

        # Initialize the positions
        node_positions = np.array([[50, 50]]).astype(np.float32)
        jammer_positions = np.array([[40, 50]]).astype(np.float32)
        user_positions = np.array([[60, 50], [60, 50]]).astype(np.float32)

        # Create the env and the agent
        env, agent = create_env_and_agent(
            node_positions, jammer_positions, user_positions, agent_type="RandomSampler"
        )
        env: gym.Env
        unwrapped_env: envs.Basic_MANETEnv = env.unwrapped
        agent: ag.AgentBase

        # Mock the agent's random generator
        class MockRandomGen:
            def multivariate_normal(self, mean, cov, size):
                return mock_positions

        agent.np_random_gen = MockRandomGen()

        assert (
            agent is not None
        ), "agent isn't correctly initialized, agent object is None"

        observedNode: comps.MANETNode = unwrapped_env.nodes[0]

        # Check that it correctly samples all three positions
        doEnvironmentSteps(env, agent, 20)
        assert np.array_equal(observedNode.pos, np.array([30, 50], dtype=np.float32))

        doEnvironmentSteps(env, agent, 10)
        assert np.array_equal(observedNode.pos, np.array([40, 50], dtype=np.float32))

        doEnvironmentSteps(env, agent, 20)
        assert np.array_equal(observedNode.pos, np.array([60, 50], dtype=np.float32))

        # Ensure that it chooses the best one, which is the last one, since it is closest to the users
        doEnvironmentSteps(env, agent, 10)
        assert np.array_equal(
            observedNode.pos, np.array([60, 50], dtype=np.float32)
        ), "should remain at the same position"


class Test_GreedyMaximizer:
    def test_SimpleCase(self):
        """Check that the GreedyMaximizer chooses the right greedy actions which should leed it to the users' positions."""
        # Initialize the positions
        node_positions = np.array([[50, 50]]).astype(np.float32)
        jammer_positions = np.array([[40, 50]]).astype(np.float32)
        user_positions = np.array([[60, 50], [60, 50]]).astype(np.float32)

        # Create the env and the agent
        env, agent = create_env_and_agent(
            node_positions,
            jammer_positions,
            user_positions,
            agent_type="GreedyMaximizer",
        )
        env: gym.Env
        unwrapped_env: envs.Basic_MANETEnv = env.unwrapped

        assert agent is not None

        observedNode: comps.MANETNode = unwrapped_env.nodes[0]

        agent: ag.AgentBase
        # Ensure that the agent keep moving
        agent.setConditionStrat(cs.SINRConditionStrategy(threshold_sinr=np.inf))

        # Expected positions (Greedy Maximizer moves twice in the best direction)
        first_eight_choices = np.array(
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]]
        ).astype(np.float32)
        first_eight_choices = first_eight_choices + observedNode.pos

        # For each position check that the GreedyMaximizer samples correctly and moves twice in to the best direction
        for chosen_position in first_eight_choices:
            relative_positions = np.array(
                [
                    [0, 1],
                    [-0.5, 0.5],
                    [-1, 0],
                    [-0.5, -0.5],
                    [0, -1],
                    [0.5, -0.5],
                    [1, 0],
                    [0.5, 0.5],
                    [0, 0],
                ]
            ).astype(np.float32)
            first_eigth_positions = relative_positions + observedNode.pos

            checkPositionsForNode(first_eigth_positions, observedNode, env, agent)
            doEnvironmentSteps(env, agent, 1)
            assert np.array_equal(observedNode.pos, chosen_position)


class Test_SpiralSearch:
    def test_SimpleCase(self):
        """
        Test that the SpiralSearch chooses the right position.
        """
        # Initialize the positions
        node_positions = np.array([[50, 50]]).astype(np.float32)
        jammer_positions = np.array([[40, 50]]).astype(np.float32)
        user_positions = np.array([[60, 50], [60, 50]]).astype(np.float32)

        # Create the env and the agent
        env, agent = create_env_and_agent(
            node_positions, jammer_positions, user_positions, agent_type="SpiralSearch"
        )
        env: gym.Env
        unwrapped_env: envs.Basic_MANETEnv = env.unwrapped

        assert agent is not None

        observedNode: comps.MANETNode = unwrapped_env.nodes[0]

        # Let it move for a while
        doEnvironmentSteps(env, agent, 100)

        # Expected position range
        expected_position_min = np.array([55, 45], dtype=np.float32)
        expected_position_max = np.array([65, 55], dtype=np.float32)

        # Final position of the node
        final_position = observedNode.pos

        # Assert that the final position is within the expected range
        np.testing.assert_array_less(
            final_position,
            expected_position_max + 1e-6,
            err_msg=f"Node's final position {final_position} is too far right or up",
        )
        np.testing.assert_array_less(
            -final_position,
            -expected_position_min + 1e-6,
            err_msg=f"Node's final position {final_position} is too far left or down",
        )

    def test_StaysInsideRadius(self):
        """
        Check that the SpiralSearch Baseline stays inside the spiral's radius.
        """
        # Initialize positions
        node_positions = np.array([[50, 50]]).astype(np.float32)
        jammer_positions = np.array([[40, 50]]).astype(np.float32)
        user_positions = np.array([[60, 50], [60, 50]]).astype(np.float32)

        # Create the env and the agent
        env, agent = create_env_and_agent(
            node_positions, jammer_positions, user_positions, agent_type="SpiralSearch"
        )
        env: gym.Env
        unwrapped_env: envs.Basic_MANETEnv = env.unwrapped

        assert agent is not None

        observedNode: comps.MANETNode = unwrapped_env.nodes[0]
        init_pos = observedNode.pos.copy()

        # Move until SpiralSearch agent settles.
        while agent.condition_strategy.is_unviable(observedNode):
            doEnvironmentSteps(env, agent, 1)

        # Ensure SpiralSearch agent hasn't left the radius of the spiral
        dist_from_origin = np.linalg.norm(init_pos - observedNode.pos)

        assert dist_from_origin <= agent.max_radius


def checkPositionsForNode(
    positions_to_check: list, node: comps.MANETNode, env: gym.Env, agent: ag.AgentBase
):
    """More elaborate version of the doEnivronmnetSteps method, which additionally for each
    step ensures that it is as expected and that the action is within the defined action space.


    Args:
        positions_to_check (list): expected positions
        node (comps.MANETNode): node to check the positions for
        env (gym.Env): environment in which to do the steps
        agent (ag.AgentBase): determines the actions
    """

    obs = env.get_wrapper_attr("observation")()
    for pos in positions_to_check:
        # Take a random action
        action, _ = agent.predict(obs)

        space = env.action_space

        # Ensure the action is valid
        assert space.contains(action)

        # Step the environment
        obs, reward, done, trunc, info = env.step(action)

        assert np.array_equal(
            node.pos, pos
        ), "node has not moved to the expected position"


def doEnvironmentSteps(env: gym.Env, agent: ag.AgentBase, numbOfSteps: int) -> None:
    """Execute x amount of steps (x = numbOfSteps) in the environment given (= env) with the predictions for the nodes made by
    the baseline agent (= agent) given.

    Args:
        env (gym.Env): the environmnent to step in
        agent (ag.AgentBase): Baseline agent to use for the determination of nodes' actions
        numbOfSteps (int): How many steps to execute
    """
    obs = env.get_wrapper_attr("observation")()

    for _ in range(numbOfSteps):
        # Take a random action
        action, _ = agent.predict(obs)

        # Ensure the action is valid
        assert env.action_space.contains(action)

        # Step the environment
        obs, reward, done, trunc, info = env.step(action)


def create_env_and_agent(
    node_positions: np.ndarray,
    jammer_positions: np.ndarray,
    user_positions: np.ndarray,
    agent_type: str = "OutAndThroughAgent",
) -> None:
    """Fixture to set up the environment and agent for testing."""
    # Create environment
    numbOfNodes = len(node_positions)
    numbOfJammers = len(jammer_positions)
    numbOfUsers = len(user_positions)

    env = envs.Basic_MANETEnv(
        numbOfNodes, numbOfJammers, numbOfUsers, allow_early_ep_finish=False
    )
    env.set_node_step_size(1)
    env.set_ep_length(10000)

    # Set node positions to own node positions
    env.node_positions = node_positions
    env.jammer_positions = jammer_positions
    env.user_positions = user_positions

    # Initialize the rest with each user in a seperate cluster.
    env.clusterSizes = np.ones(numbOfUsers, dtype=int)
    env.createComponents()
    env.initializeDemand()
    env.initializeSendersAndReceivers()

    # Indices
    env.user_to_idx = {user: idx for idx, user in enumerate(env.users)}
    # Activate the jammers
    for jammer in env.jammers:
        jammer: comps.Jammer
        jammer.startJamming()

    env.network = envs.Network(env.nodes, env.jammers, env.users)

    # Wrap the environment
    env = ut.wrapEnv(
        env=env,
        rew_wrapper=rw.EvaluationReward,
        obs_wrapper=ow.VS_COW,
        use_monitor=False,
    )

    if agent_type == "OutAndThroughAgent":
        agent = ag.OutAndThroughAgent(env, max_timeout=0)
    elif agent_type == "RandomAgent":
        agent = ag.RandomAgent(env, max_timeout=0)
    elif agent_type == "RandomSampler":
        agent = ag.RandomSampler(env, max_timeout=0)
    elif agent_type == "GreedyMaximizer":
        agent = ag.GreedyMaximizer(env, max_timeout=0)
    elif agent_type == "SpiralSearch":
        agent = ag.SpiralSearch(env, max_timeout=0)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return env, agent


def test_getStepsTo_method():
    """
    Test that given a destination and the current position the correctly scaled steps are given to reach it by the getStepsTo_method
    """
    current_pos = np.array([0, 0])
    dest = np.array([3, 4])
    expected_steps = np.array(
        [[0.6, 0.8], [0.6, 0.8], [0.6, 0.8], [0.6, 0.8], [0.6, 0.8]], dtype=np.float32
    )

    result = ag.getStepsTo(current_pos, dest)
    np.testing.assert_array_almost_equal(result, expected_steps)
