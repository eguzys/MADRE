import os
import shutil
import pytest
import yaml

import Utils.training_utils as ut


@pytest.fixture(scope="function")
def setup_and_teardown_logs():
    # Determine the directory of the current test file
    test_dir = os.path.dirname(__file__)
    base_dir = os.path.join(test_dir, "out_test/models")

    # Change the working directory to the test file's directory
    original_cwd = os.getcwd()
    os.chdir(test_dir)

    # Setup: Create test directories and files
    dirs_to_create = [
        f"{base_dir}/30_12/PPO_3_1_3_20k_capacities",
        f"{base_dir}/30_12/PPO_3_1_3_50k_capacities",
        f"{base_dir}/30_12/PPO_3_1_3_no_models",
    ]
    dirs_to_create = ut.convertPathsToPyPath(*dirs_to_create)
    ut.create_dirs(*dirs_to_create)

    # Create test files
    open(
        f"{base_dir}/30_12/PPO_3_1_3_20k_capacities/01_training_config.yaml", "a"
    ).close()
    open(
        f"{base_dir}/30_12/PPO_3_1_3_20k_capacities/rl_model_500000_steps.zip", "a"
    ).close()
    open(
        f"{base_dir}/30_12/PPO_3_1_3_20k_capacities/rl_model_1000000_steps.zip", "a"
    ).close()
    open(
        f"{base_dir}/30_12/PPO_3_1_3_20k_capacities/rl_model_2000000_steps.zip", "a"
    ).close()
    open(
        f"{base_dir}/30_12/PPO_3_1_3_50k_capacities/rl_model_1000000_steps.zip", "a"
    ).close()

    yield

    # Teardown: Restore the original working directory and remove test files
    os.chdir(original_cwd)
    shutil.rmtree(os.path.dirname(base_dir))


class Test_getAgent_Paths:
    def test_findMultipleUnderDateFolder(self, setup_and_teardown_logs):
        patterns = ["out_test/models/30_12/*"]
        paths_of_agents = ut.findAgentsMatchingPattern(*patterns)
        expected_paths_of_agents = sorted(
            [
                "out_test/models/30_12/PPO_3_1_3_20k_capacities/rl_model_2000000_steps.zip",
                "out_test/models/30_12/PPO_3_1_3_50k_capacities/rl_model_1000000_steps.zip",
            ]
        )

        assert sorted(paths_of_agents) == expected_paths_of_agents

    def test_findSpecificAgentAnyModel(self, setup_and_teardown_logs):
        patterns = ["out_test/models/30_12/PPO_3_1_3_20k_capacities/*"]
        paths_of_agents = ut.findAgentsMatchingPattern(*patterns)
        expected_paths_of_agents = [
            "out_test/models/30_12/PPO_3_1_3_20k_capacities/rl_model_2000000_steps.zip"
        ]
        assert paths_of_agents == expected_paths_of_agents

    def test_findExactModels(self, setup_and_teardown_logs):
        patterns = [
            "out_test/models/30_12/PPO_3_1_3_20k_capacities/rl_model_500000_steps.zip",
            "out_test/models/30_12/PPO_3_1_3_50k_capacities/rl_model_1000000_steps.zip",
        ]
        paths_of_agents = ut.findAgentsMatchingPattern(*patterns)
        expected_paths_of_agents = [
            "out_test/models/30_12/PPO_3_1_3_20k_capacities/rl_model_500000_steps.zip",
            "out_test/models/30_12/PPO_3_1_3_50k_capacities/rl_model_1000000_steps.zip",
        ]
        assert sorted(paths_of_agents) == sorted(expected_paths_of_agents)

    def test_noModelsFound(self, setup_and_teardown_logs):
        patterns = ["out_test/models/30_12/PPO_3_1_3_no_models/*"]
        paths_of_agents = ut.findAgentsMatchingPattern(*patterns)
        expected_paths_of_agents = []
        assert paths_of_agents == expected_paths_of_agents

    def test_SpecifiedStepsButNotAgent(self, setup_and_teardown_logs):
        patterns = [
            "out_test/models/30_12/*/rl_model_1000000_steps.zip",
        ]
        paths_of_agents = ut.findAgentsMatchingPattern(*patterns)
        expected_paths_of_agents = [
            "out_test/models/30_12/PPO_3_1_3_20k_capacities/rl_model_1000000_steps.zip",
            "out_test/models/30_12/PPO_3_1_3_50k_capacities/rl_model_1000000_steps.zip",
        ]
        assert sorted(paths_of_agents) == sorted(expected_paths_of_agents)


class Test_getFileNamesFromPaths:

    def test_simpleCase(self):
        paths = [
            "some_dir/some_subdir/some_file.csv",
            "other_dir/other_subdir/other_file.csv",
        ]
        expected_filenames = ["some_file", "other_file"]

        found_filenames = ut.getFileNamesFromPaths(*paths)

        assert sorted(expected_filenames) == sorted(found_filenames)


@pytest.fixture(scope="function")
def setup_and_teardown_agents():
    # Determine the directory of the current test file
    test_dir = os.path.dirname(__file__)
    base_dir = os.path.join(test_dir, "out_test/models")

    # Change the working directory to the test file's directory
    original_cwd = os.getcwd()
    os.chdir(test_dir)

    # Setup: Create test directories and files
    dirs_to_create = [
        f"{base_dir}/29_02",
        f"{base_dir}/30_02",
        f"{base_dir}/31_02",
    ]
    dirs_to_create = ut.convertPathsToPyPath(*dirs_to_create)
    ut.create_dirs(*dirs_to_create)

    agent_folders = [
        "PPO_4_0_2_30k_capacity",
        "PPO_4_0_2_30k_no_capacity",
        "A2C_4_0_2_30k_capacity",
        "A2C_4_0_2_30k_no_capacity",
    ]

    for date_dir in dirs_to_create:
        for agent_folder in agent_folders:
            agent_path = os.path.join(date_dir, agent_folder)
            ut.create_dirs(agent_path)
            # Create empty zip files
            for i in range(1, 4):
                open(
                    os.path.join(agent_path, f"rl_model_{i*1000000}_steps.zip"), "a"
                ).close()

    # Teardown: Restore the original working directory and remove test files
    yield
    os.chdir(original_cwd)
    shutil.rmtree(os.path.dirname(base_dir))


class Test_loadAgentFolders:
    def test_loadFromDateDirs_simple(self, setup_and_teardown_agents):
        patterns = ["*/models/29_02"]

        expected_agent_folders = [
            "PPO_4_0_2_30k_capacity",
            "PPO_4_0_2_30k_no_capacity",
            "A2C_4_0_2_30k_capacity",
            "A2C_4_0_2_30k_no_capacity",
        ]

        expected_agent_folder_paths = [
            "out_test/models/29_02/" + folder_name
            for folder_name in expected_agent_folders
        ]

        agent_folder_paths = ut.findAgentFoldersPaths(*patterns)

        assert sorted(expected_agent_folder_paths) == sorted(agent_folder_paths)

    def test_loadFromDateDirs_asPattern(self, setup_and_teardown_agents):
        patterns = ["*/models/29_02/*"]

        expected_agent_folders = [
            "PPO_4_0_2_30k_capacity",
            "PPO_4_0_2_30k_no_capacity",
            "A2C_4_0_2_30k_capacity",
            "A2C_4_0_2_30k_no_capacity",
        ]

        expected_agent_folder_paths = [
            "out_test/models/29_02/" + folder_name
            for folder_name in expected_agent_folders
        ]

        agent_folder_paths = ut.findAgentFoldersPaths(*patterns)

        assert sorted(expected_agent_folder_paths) == sorted(agent_folder_paths)

    def test_loadFromAgentDirs_simple(self, setup_and_teardown_agents):
        patterns = ["*/models/29_02/PPO_4_0_2_30k_capacity"]

        expected_agent_folder_paths = ["out_test/models/29_02/PPO_4_0_2_30k_capacity"]

        agent_folder_paths = ut.findAgentFoldersPaths(*patterns)

        assert sorted(expected_agent_folder_paths) == sorted(agent_folder_paths)

    def test_loadFromAgentDirs_multiple(self, setup_and_teardown_agents):
        patterns = [
            "*/models/29_02/PPO_4_0_2_30k_capacity",
            "*/models/29_02/PPO_4_0_2_30k_no_capacity",
        ]

        expected_agent_folder_paths = [
            "out_test/models/29_02/PPO_4_0_2_30k_capacity",
            "out_test/models/29_02/PPO_4_0_2_30k_no_capacity",
        ]

        agent_folder_paths = ut.findAgentFoldersPaths(*patterns)

        assert sorted(expected_agent_folder_paths) == sorted(agent_folder_paths)

    def test_loadFromMultipleDateDirs(self, setup_and_teardown_agents):
        patterns = [
            "*/models/29_02/*",
            "*/models/30_02/*",
        ]

        expected_agent_folders = [
            "PPO_4_0_2_30k_capacity",
            "PPO_4_0_2_30k_no_capacity",
            "A2C_4_0_2_30k_capacity",
            "A2C_4_0_2_30k_no_capacity",
        ]

        expected_agent_folder_paths = [
            f"out_test/models/{date}/{folder_name}"
            for date in ["29_02", "30_02"]
            for folder_name in expected_agent_folders
        ]

        agent_folder_paths = ut.findAgentFoldersPaths(*patterns)

        assert sorted(expected_agent_folder_paths) == sorted(agent_folder_paths)

    def test_loadFromDateDirs_multipleAsPattern(self, setup_and_teardown_agents):
        patterns = ["*/models/3?_02/"]

        expected_agent_folders = [
            "PPO_4_0_2_30k_capacity",
            "PPO_4_0_2_30k_no_capacity",
            "A2C_4_0_2_30k_capacity",
            "A2C_4_0_2_30k_no_capacity",
        ]

        expected_agent_folder_paths = [
            f"out_test/models/{date}/{folder_name}"
            for date in ["30_02", "31_02"]
            for folder_name in expected_agent_folders
        ]

        agent_folder_paths = ut.findAgentFoldersPaths(*patterns)

        assert sorted(expected_agent_folder_paths) == sorted(agent_folder_paths)


@pytest.fixture(scope="function")
def setup_teardown_configs():
    """
    Pytest fixture to set up nested and flat config files for testing.
    """
    # Determine the directory of the current test file
    test_dir = os.path.dirname(__file__)
    config_dir = os.path.join(test_dir, "test_configs")

    # Change the working directory to the test file's directory
    original_cwd = os.getcwd()
    os.chdir(test_dir)
    os.makedirs(config_dir, exist_ok=True)

    flat_config = {
        "RL_model_name": "PPO",
        "allow_early_ep_finish": False,
        "attackerModel": "Static_ClusterJammers",
        "env": "Static_MANETEnv",
        "ep_length": 5000,
        "jammersSpawnNextToUsers": True,
        "log_path": "out/logs/26_01",
        "model_path": "out/models/26_01/PPO_1_0_2_5k",
        "networkConnectedAtStart": False,
        "numbOfJammers": 0,
        "numbOfNodes": 1,
        "numbOfUsers": 2,
        "obs_config": {
            "capacity_matrix": False,
            "nodes": {
                "interference": True,
                "interference_diff": False,
                "lastAction": False,
                "receivedSignal": True,
            },
            "users": {
                "destination": True,
                "interference": True,
                "receivedSignal": True,
            },
        },
        "obs_wrapper": "VS_iCOW",
        "render": False,
        "rew_wrapper": "SquaredReward",
        "save_freq": 5000000,
        "seed": 3,
        "stat_size": 10,
        "steps_till_jammer_active": 0,
        "tb_logname": "PPO_1_0_2_5k",
        "timesteps": 100000000,
    }

    ut.document_as_nested_config(config_dir, "nested_config.yaml", **flat_config)
    ut.document_config(config_dir, "flat_config.yaml", **flat_config)

    yield
    os.chdir(original_cwd)
    shutil.rmtree(config_dir)


class Test_isNested:

    def test_nestedCase(self, setup_teardown_configs):

        config = getConfig("test_configs", "nested_config.yaml")

        is_nested = ut.is_nested_config(config)

        assert is_nested

    def test_notNestedCase(self, setup_teardown_configs):
        config = getConfig("test_configs", "flat_config.yaml")

        is_nested = ut.is_nested_config(config)

        assert not is_nested


def getConfig(config_dir: str, config_name: str) -> dict:
    """
    Load the training configuration from a YAML file.

    Args:
        config_dir (str): Path to the directory containing the configuration file.
        config_name (str): Name of the configuration file.

    Returns:
        dict: Loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    config_path = os.path.join(config_dir, config_name)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
