"""
### Overview:
This script trains an agent using a configuration file. You don't need to specify all parameters
in the configuration file; unspecified values will default to their predefined values.

### Default Parameters:
1. **Storing and Logging**:
   - `model_dir`: "out/models" (Directory to save the trained model)
   - `log_dir`: "out/logs" (Directory to save logs)
   - `log_name`: "" (Name of the log file)

2. **Model**:
   - `RL_model_name`: "PPO" (Reinforcement Learning model to use)

3. **Environment**:
   - `env`: "Static_MANETEnv" (Environment to train the agent in)
   - `render`: False (Whether to render the environment during training)

4. **Environment Parameters**:
   - `numbOfNodes`: 5 (Number of MANET nodes)
   - `numbOfJammers`: 1 (Number of jammers)
   - `numbOfUsers`: 10 (Number of users)
   - `seed`: 9 (Random seed for reproducibility)
   - `networkConnectedAtStart`: False (Whether the network is connected at the start)
   - `jammersSpawnNextToUsers`: True (Whether jammers spawn near users)
   - `allow_early_ep_finish`: True (Allow early episode termination)
   - `step_size`: 1 (The maximum step size for each nodes action, meaning a_i \in step_size * [-1,1]^2)


5. **Attacker Model**:
   - `attackerModel`: "Static_ClusterJammers" (Type of attacker model)
   - `steps_till_jammer_active`: 0 (Steps before jammers become active)
   - `step_jammers_start_moving`: 0 (Step from which on the jammers start movig to position themselves.)

6. **Observation Wrapper**:
   - `obs_wrapper`: "VS_iCOW" (Wrapper for observations)
   - `obs_config`: None (Configuration for the observation wrapper)

7. **Reward Wrapper**:
   - `rew_wrapper`: "SquaredReward" (Wrapper for rewards)

8. **Training Parameters**:
   - `timesteps`: 30,000,000 (Total training timesteps)
   - `ep_length`: 10,000 (Maximum episode length)
   - `save_freq`: 500,000 (Frequency of saving the model)
   - `stat_size`: 10 (Size of statistics buffer)

### Usage:
- Specify the path to the configuration file in the `config_path` parameter.
- Call the `train_agent_from_config` function to start training.
"""

# To train the agent
import train_MANETAgent as trainer

# * Train from the config
trainer.train_agent_from_config(config_path="src/examples/train/example_train_config.yaml")
