"""
This function trains a reinforcement learning agent in a MANET environment with configurations specified by the parameters. Logging and saving occur based on provided directories and frequencies.

### How It Works:
1. **Parameter Resolution**:
- The function accepts parameters for environment setup, RL model configuration, logging, and training.
- If certain parameters (e.g., `env`, `attackerModel`, `obs_wrapper`, `rew_wrapper`) are provided as strings, they are resolved to their respective class constructors using utility functions.

2. **Environment Setup**:
- The environment is created and wrapped using the `createAndWrapEnv` utility function. This includes:
    - Setting up the MANET environment with the specified number of nodes, jammers, and users.
    - Applying observation and reward wrappers.
    - Configuring attacker models and other environment parameters.

3. **Logging and Storage**:
- The `setDirectories` utility function sets up directories for storing the model, logs, and TensorBoard files.
- A custom callback (`CheckpointCallback`) is used to save the model periodically during training.

4. **Agent Initialization**:
- The RL agent is initialized using the specified RL model (e.g., PPO) and the wrapped environment.
- The agent's statistics buffer size and random seed are also configured.

5. **Configuration Storage**:
- The training configuration is stored as a YAML file (`01_training_config.yaml`) in the model directory. This includes all parameters used for training, ensuring reproducibility.

6. **Training**:
- The agent is trained using the `learn` method, which runs for the specified number of timesteps.
- Progress is logged to TensorBoard, and the model is saved at regular intervals.

### Examples:
1. **Model Storage**:
- If `model_dir="out/models"`, `log_name="test"`, `RL_model_name="PPO"`, `numbOfNodes=5`, `numbOfJammers=2`, `numbOfUsers=10`, and `ep_length=10000`:
    - The trained model will be stored in:
    ```
    out/models/<date>/PPO_5_2_10_10k_test/
    ```
    - The model files will include checkpoints saved at intervals (e.g., `rl_model_500000_steps.zip`).

2. **Log Storage**:
- If [log_dir="out/logs"], [log_name="test"], [RL_model_name="PPO"], [numbOfNodes=5], [numbOfJammers=2], [numbOfUsers=10], and [ep_length=10000]:
    - The logs will be stored in:
    ```
    out/logs/<date>/PPO_5_2_10_10k_test_1/
    ```
    - TensorBoard logs will also be saved in this directory for visualization.

3. **Configuration Storage**:
- The training configuration will be saved as:
    ```
    out/models/<date>/PPO_5_2_10_10k_test/01_training_config.yaml
    ```
- This file contains all parameters used for training, ensuring reproducibility.

"""

# * Import needed libs
# To train the agent
import train_MANETAgent as trainer

# Own classes
import MANET.ObservationWrappers as ow
import MANET.Environments as envs
import MANET.RewardWrappers as rw
import MANET.Attackers as at

# * Choose env and set env parameters
# Choose env
env = envs.Dynamic_MANETEnv

# Set parameters
env_parameters = {
    "numbOfNodes": 4,
    "numbOfJammers": 2,
    "numbOfUsers": (2, 9),
    "seed": 2,
    "networkConnectedAtStart": False,
    "jammersSpawnNextToUsers": True,
    "allow_early_ep_finish": True,
    "step_size": 2,
}

# * Choose attackermodel
attackerModel = [at.Static_ClusterJammers, at.Static_TrafficJammers]

# * Choose oberservation wrapper
# Create obswrapper and wrap env
obs_wrapper = ow.VS_COW
# Choose parameters to use for observation wrapper
obs_config = {
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

# * Choose reward function wrapper
rew_wrapper = rw.LinearReward

# * Train agent with you choosen parameters
# Set training parameters
training_params = {
    "timesteps": 80_000_000,
    "ep_length": 15_000,
    "save_freq": 1_000_000,
    "routing_func": "sequential_max_flow",
    "step_jammers_start_moving": 50,
    "steps_till_jammer_active": 150,
}

trainer.train_agent(
    RL_model_name="PPO",
    log_name="test",
    env=env,
    render=True,
    **env_parameters,
    attackerModel=attackerModel,
    obs_wrapper=obs_wrapper,
    obs_config=obs_config,
    rew_wrapper=rew_wrapper,
    **training_params,
)
