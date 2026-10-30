import os
import shutil
import pytest
import pandas as pd
import numpy as np

from visualize_MANETAgents import (
    findLogFilesFromPattern,
    find_stabilizing_index,
    DataVisualizer,
)
import Utils.training_utils as ut


@pytest.fixture
def initDataVisualizer() -> DataVisualizer:
    log_patterns = [
        "out_test/evaluations/3_0_2_throughput/*",
        "out_test/evaluations/3_1_3_throughput/*",
        "out_test/evaluations/4_0_4_throughput/*",
    ]
    return DataVisualizer(*log_patterns)


@pytest.fixture(scope="function")
def setup_and_teardown():
    # Determine the directory of the current test file
    test_dir = os.path.dirname(__file__)
    base_dir = os.path.join(test_dir, "out_test/evaluations")

    # Change the working directory to the test file's directory
    original_cwd = os.getcwd()
    os.chdir(test_dir)

    # Setup: Create test directories and files
    dirs_to_create = [
        f"{base_dir}/3_0_2_throughput",
        f"{base_dir}/3_1_3_throughput",
        f"{base_dir}/4_0_4_throughput",
    ]
    dirs_to_create = ut.convertPathsToPyPath(*dirs_to_create)
    ut.create_dirs(*dirs_to_create)

    # Create test files and add example data
    example_data = "ep_step,episode,value_name,value\n1,0,throughput,2.0\n2,0,throughput,4.0\n3,0,throughput,3.0\n\n1,1,throughput,5.0\n2,1,throughput,4.0\n3,1,throughput,3.0\n"
    files_to_create = [
        f"{base_dir}/3_0_2_throughput/PPO_3_0_2_20k_capacity.csv",
        f"{base_dir}/3_0_2_throughput/PPO_3_0_2_20k_no_capacity.csv",
        f"{base_dir}/3_0_2_throughput/A2C_3_0_2_20k_no_capacity.csv",
        f"{base_dir}/3_1_3_throughput/GreedyMaximizer.csv",
        f"{base_dir}/3_1_3_throughput/A2C_3_1_3_20k_no_capacity.csv",
    ]

    for file_path in files_to_create:
        with open(file_path, "w") as file:
            file.write(example_data)

    yield

    # Teardown: Restore the original working directory and remove test files
    os.chdir(original_cwd)
    shutil.rmtree(os.path.dirname(base_dir))


class Test_findLogPathsFromPattern:
    def test_simpleCase(self, setup_and_teardown) -> None:
        pattern = ["**/3_0_2_throughput/*"]
        expected_paths = [
            "out_test/evaluations/3_0_2_throughput/PPO_3_0_2_20k_capacity.csv",
            "out_test/evaluations/3_0_2_throughput/PPO_3_0_2_20k_no_capacity.csv",
            "out_test/evaluations/3_0_2_throughput/A2C_3_0_2_20k_no_capacity.csv",
        ]

        found_paths = findLogFilesFromPattern(*pattern)

        assert sorted(expected_paths) == sorted(found_paths)

    def test_advancedCase(self, setup_and_teardown) -> None:
        pattern = ["**/3_*_throughput/*"]
        expected_paths = [
            "out_test/evaluations/3_0_2_throughput/PPO_3_0_2_20k_capacity.csv",
            "out_test/evaluations/3_0_2_throughput/PPO_3_0_2_20k_no_capacity.csv",
            "out_test/evaluations/3_0_2_throughput/A2C_3_0_2_20k_no_capacity.csv",
            "out_test/evaluations/3_1_3_throughput/GreedyMaximizer.csv",
            "out_test/evaluations/3_1_3_throughput/A2C_3_1_3_20k_no_capacity.csv",
        ]

        found_paths = findLogFilesFromPattern(*pattern)

        assert sorted(expected_paths) == sorted(found_paths)

    def test_edgeCaseNoCSV(self, setup_and_teardown) -> None:
        pattern = ["**/4_0_4_throughput/*"]
        expected_paths = []

        found_paths = findLogFilesFromPattern(*pattern)

        assert sorted(expected_paths) == sorted(found_paths)


class Test_DataVisualizer:
    def test_loadLogs(self, setup_and_teardown, initDataVisualizer):
        dv: DataVisualizer = initDataVisualizer

        expected_paths = [
            "out_test/evaluations/3_0_2_throughput/PPO_3_0_2_20k_capacity.csv",
            "out_test/evaluations/3_0_2_throughput/PPO_3_0_2_20k_no_capacity.csv",
            "out_test/evaluations/3_0_2_throughput/A2C_3_0_2_20k_no_capacity.csv",
            "out_test/evaluations/3_1_3_throughput/GreedyMaximizer.csv",
            "out_test/evaluations/3_1_3_throughput/A2C_3_1_3_20k_no_capacity.csv",
        ]

        assert sorted(expected_paths) == sorted(dv.log_loader.log_paths)


import matplotlib.pyplot as plt


class Test_helperMethods:
    # Test 1: Sinusoidal Function with Noise
    def test_sinusoidal_with_noise(self):
        # Create a sinusoidal function with noise
        x = np.linspace(0, 100, 100)  # 100 points
        throughput = (
            5 + np.sin(x) + np.random.normal(0, 0.1, 100)
        )  # Sinusoidal signal with noise
        throughput_series = pd.Series(throughput)

        # Run the stability detection function
        start_step = find_stabilizing_index(throughput_series, epsilon_factor=0.1)

        # Check that the interval is found and its length is reasonable (e.g., near the end)
        assert start_step is not None, "No stability interval found!"

    # Test 2: Logarithmic-ish Function with Noise
    def test_logarithmic_with_noise(self):
        # Create a logarithmic function with noise that stabilizes around x = 4
        x = np.linspace(1, 1e9, 100)  # Avoid log(0) by starting from 1
        throughput = np.log(x) + np.random.normal(
            0, 0.01, 100
        )  # Logarithmic with noise
        throughput_series = pd.Series(throughput)

        # Run the stability detection function
        start_step = find_stabilizing_index(throughput_series, epsilon_factor=0.1)

        assert start_step is not None
