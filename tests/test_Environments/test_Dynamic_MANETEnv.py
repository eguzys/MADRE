# Need for testing
from unittest.mock import patch, MagicMock, Mock
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


def create_non_wrapped_env(
    numbOfNodes=5, numbOfJammers=2, numbOfUsers=10
) -> envs.Dynamic_MANETEnv:
    env = envs.Dynamic_MANETEnv(
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        render_mode=None,
        seed=42,
    )
    env.reset(seed=42)
    return env


def create_wrapped_env(numbOfNodes=5, numbOfJammers=2, numbOfUsers=10) -> gym.Env:
    env = envs.Dynamic_MANETEnv(
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        render_mode=None,
        seed=42,
    )
    env.reset(seed=42)
    env = rw.MovementCost(env)
    env = ow.VS_COW(env)
    return env


@pytest.fixture
def wrapped_env() -> gym.Env:
    env = create_wrapped_env()

    return env


@pytest.fixture
def non_wrapped_env() -> envs.Dynamic_MANETEnv:
    env = create_non_wrapped_env()

    return env


@pytest.fixture
def non_wrapped_uninit_env() -> envs.Dynamic_MANETEnv:
    with patch("MANET.Environments.Dynamic_MANETEnv.reset") as reset_mock:
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
def wrapped_uninit_env() -> envs.Dynamic_MANETEnv:
    with patch("MANET.Environments.Dynamic_MANETEnv.reset") as reset_mock:
        reset_mock.return_value = (
            None,
            None,
        )  # Mock reset to prevent full initialization

        # Initialize the environment but without `reset` running fully
        env = (
            create_wrapped_env()
        )  # Or use `create_non_wrapped_env` if you want an unwrapped environment

        return env


class Test_createDestProbMatrix:
    def test_createDestProbMatrix_shape(self, non_wrapped_env):
        env: envs.Dynamic_MANETEnv = non_wrapped_env
        env.createDestProbMatrix()
        assert env.destProbabilities.shape == (env.numbOfUsers, env.numbOfUsers)

    def test_createDestProbMatrix_no_self_prob(self, non_wrapped_env):
        env: envs.Dynamic_MANETEnv = non_wrapped_env
        env.createDestProbMatrix()
        assert np.all(np.diag(env.destProbabilities) == 0)

    def test_createDestProbMatrix_row_are_normalized(self, non_wrapped_env):
        env: envs.Dynamic_MANETEnv = non_wrapped_env
        env.createDestProbMatrix()
        row_sums = env.destProbabilities.sum(axis=1)
        assert np.allclose(row_sums, 1)

    def test_createDestProbMatrix_distribution(self, non_wrapped_env):
        env: envs.Dynamic_MANETEnv = non_wrapped_env
        env.createDestProbMatrix()
        alpha = 1
        beta = 4
        expected_mean = alpha / (alpha + beta)
        actual_mean = np.mean(env.destProbabilities)
        assert np.isclose(actual_mean, expected_mean, atol=0.1)


class Test_simulateTrafficPartnerChoice:
    @patch("MANET.Environments.Dynamic_MANETEnv.simulateTransmittingChoices")
    def test_simulateTrafficPartnerChoice(self, mock_transmiss_choices: Mock):
        env: envs.Dynamic_MANETEnv = create_non_wrapped_env(
            numbOfNodes=5, numbOfJammers=0, numbOfUsers=3
        )

        # Mock simulateTransmittingChoices to set user_transmitter_flags
        def mock_simulate_transmitting_choices():
            env.user_transmitter_flags = np.ones(env.numbOfUsers, dtype=int)

        mock_transmiss_choices.side_effect = mock_simulate_transmitting_choices

        continue_sending_value = 3
        answer_value = 1
        norm_factor = continue_sending_value + answer_value

        # User 1 sends to User2, User 2 to User 3, User 3 to User 1
        env.destProbabilities = np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32
        )

        env.simulateTrafficPartnerChoice()

        user1, user2, user3 = env.users

        # Check destinations chosen correctly
        assert user1.dest == user2
        assert user2.dest == user3
        assert user3.dest == user1

        # Check probabilities updated correctly
        expected_new_destProbs = (
            np.array(
                [
                    [0, continue_sending_value, answer_value],
                    [answer_value, 0, continue_sending_value],
                    [continue_sending_value, answer_value, 0],
                ]
            )
            / norm_factor
        )

        assert np.allclose(env.destProbabilities, expected_new_destProbs, rtol=1e-4)


class Test_createOfferedLoadProbMatrix:
    # not test need, very simple
    pass


class Test_initializeSendingData:
    # not worth testing, rather simple
    pass


class Test_simulateOfferedLoadChoice:
    def test_destChosenCorrectly(self):
        env: envs.Dynamic_MANETEnv = create_non_wrapped_env(2, 0, 3)

        env.user_sending_data_type = np.array([0, 0, 1])

        env.offLoadProbabilities = np.array([[1, 0], [1, 0]])

        env.simulateOfferedLoadChoice()

        expected_new_data_types = np.array([0, 0, 0])
        assert np.array_equal(env.user_sending_data_type, expected_new_data_types)
