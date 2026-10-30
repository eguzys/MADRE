# Installation

The project is configured for Python 3.11, so any version of 3.11.x. Make sure to use the correct Python interpreter. The dependencies are managed with Poetry. Either use a pip version that supports PEP 517 and PEP 518 to install the project's dependencies via `pip install .` or install Poetry via pip `pip install poetry` and then run `poetry install` in the root folder. Using Poetry is the safer option, as using `pip install .` can cause trouble with recognizing certain modules or packages. 

If you are interested in running the `MkDocs`-page or the tests that were created with `pytest`, use `poetry` to install the dependencies, which will ensure that all the developer dependencies are installed as well.

To use the OptimumFinder (although is not a very important part of the project) one must install the `ipopt` optimizer. If you already use conda for your python environment, run the following command from the same conda env:
```bash
conda install -c conda-forge ipopt
```

# MANETNodesRelocation - Tutorial

Before running any of the code ensure that your system path is set to src otherwise modules are not correctly imported.


```python
import sys
import os

# Determine the path to the src directory
src_path = os.path.join(os.getcwd(), 'src')

# Add the src directory to the Python path
if src_path not in sys.path:
    sys.path.append(src_path)
```

## 1. Introduction

This notebook demonstrates how to work with the MANETNodesRelocation framework for training, retraining, filtering, and evaluating reinforcement learning agents in Mobile Ad-hoc Network (MANET) environments. The workflow typically follows this order:

1. Train an agent on a MANET environment
2. Optionally retrain the agent to improve performance
3. Filter the saved models to select the best performers
4. Evaluate the filtered models against baseline strategies

An important thing to consider is that in our simulation, what in the thesis is defined as a unit square $[0,1]^2$ is scaled to $[0,\text{SIZE}]^2$, with $\text{SIZE}=100$ per default.

## 2. Training Agents

### 2.1 Setup for Training

Import the trainer module


```python
import train_MANETAgent as trainer
```

You can train an agent using either:
- A configuration file (YAML format)
- Direct function calls with parameters

Why use the YAML format?

- Some parameters are nested anyhow, hence a format that allows for nested data is needed.
- YAML further is very human readable.

### 2.2 Training with Configuration File

#### Example Configuration File
An example configuration file can be found at `src/examples/train/example_train_config.yaml`. The file specifies various training parameters. If any parameter is not defined, it will use a default value. So, one needn't specify all params, but non-specified parameters will revert to their default value. Below you find the key configuration parameters. To see all parameters that can be defined, see under [8.1](#81-all-training-config-file-options).

#### Key Configuration Parameters
| Category | Parameter | Default Value | Description |
|----------|-----------|---------------|-------------|
| **Storing and Logging** | `model_dir` | "out/models" | Directory to save the trained model |
|  | `log_dir` | "out/logs" | Directory to save logs |
|  | `log_name` | "" | Name of the log file |
| **Model** | `RL_model_name` | "PPO" | Reinforcement Learning model type |
| **Environment** | `env` | "Static_MANETEnv" | Environment to train in |
| **Environment Parameters** | `numbOfNodes` | 5 | Number of MANET nodes |
|  | `numbOfJammers` | 1 | Number of jammers |
|  | `numbOfUsers` | 10 | Number of users |
| **Training Parameters** | `timesteps` | 30,000,000 | Total training timesteps |
|  | `ep_length` | 10,000 | Maximum episode length |


#### Special Parameters

The number of nodes `numbOfNodes` is always fix for each RL agent as the observation space is fixed size. But, if the observation space is sensibly configured (not including user observations or the capacity matrix) through the observation config, then the agent can be trained on variable numbers of users (and obviously jammers). However, it might be annoying to retrain an agent for each specific number of users `numbOfUsers` or `jammers`. Thus, in training we allow that one can set the `numbOfUsers` and `numbOfJammers` as an interval from which at the start of each episode the actual number of users and jammers are informly randomly sampled, instead of a scalar. The interval is given as a `list` or a `tuple`.

A similar issue arises, when training an RL agent for different attacker models. Thus, one can also provide a `list` or a `tuple` with the attacker models either their name as a string or the attacker model objects and at the start of each episode the attacker model is uniformly randomly sampled from that list.

#### Running Training with Config File


```python
trainer.train_agent_from_config(
    config_path="src/examples/train/example_train_config.yaml"
)
```

### 2.3 Training with Direct Function Call


```python
trainer.train_agent(
    # storing and logging directories and names
    model_dir = "out/models",
    log_dir = "out/logs",
    log_name = "",
    # model
    RL_model_name = "PPO",  # Options PPO, A2C, SAC, TD3, DDPG
    # env
    env = "Static_MANETEnv",
    # whether to render env
    render = False,
    # env parameters
    numbOfNodes = 5,
    numbOfJammers = 1,
    numbOfUsers = [2,21], # samples the interval [2,21) but only integers
    seed = 9,
    networkConnectedAtStart = False,
    jammersSpawnNextToUsers = False,
    allow_early_ep_finish = True,
    routing_func = "sequential_max_flow",
    step_size = 2,
    # attacker model
    attackerModel = ["Static_ClusterJammers", "Static_TrafficJammers"],
    steps_till_jammer_active = 0,
    step_jammers_start_moving = 0,
    # obswrapper
    obs_wrapper = "VS_iCOW",
    # obs config
    obs_config = None,
    # rew wrapper
    rew_wrapper = "SquaredReward",
    # training params
    timesteps = 30_000_000,
    ep_length = 10_000,
    save_freq = 500_000,
    stat_size = 10,
    )
```

### 2.4 Key Component Options

#### Available Environments
- `Static_MANETEnv`: Basic environment with static users and traffic
- `Less_Static_MANETEnv`: Adds slow user movement in clusters according to adapted RPGM model
- `Dynamic_MANETEnv`: Adds realistic traffic patterns and changing destinations on top of the movement

Special parameters:

- networkConnectedAtStart `bool`: One can choose whether the Nodes and Users should be distributed such that the Nodes are connected to each other or, in other words, form a connected graph.
- jammersSpawnNextToUsers `bool`: Whether Jammers are initiated next to a User
- steps_till_jammers_active `int`: The step from which on the jammers activate and if static become stationary.
- step_jammers_start_moving `int`: The step from which on the jammers start to find a good position to jam.
- allow_early_ep_finish `bool`: Whether an epsiode can finish before it has run through the specified `ep_length` either through achieving the maximum possible throughput or not making any advances, meaning that either its throughput barely changes of a fixed interval or the nodes only barely move over a the same fixed interval.

#### Attacker Models

- `Static_GreedyJammers` / `Moving_GreedyJammers`: Uses signal strength
- `Static_ClusterJammers` / `Moving_ClusterJammers`: Targets user clusters
- `Static_TrafficJammers` / `Moving_TrafficJammers`: Targets high-traffic nodes

#### Observation Wrappers

- **Fixed-sized observation spaces:** These always maintain the size of the full observation space. This allows flexibility in including or excluding variables without altering the observation space size. It is particularly useful for retraining with previously excluded variables.
  
- **Variable-sized observation spaces:** These reduce the observation space to the minimum required size, including only the selected observations. While more efficient, they do not allow for easy expansion without redefining the space.

Both fixed-sized and variable-sized spaces maintain constant dimensions in practice, adhering to the requirements of StableBaselines3.

**Invariable Versions**

To handle symmetries in the observation space, invariable versions of these wrappers are introduced. Symmetry elimination ensures that the agent does not need to learn redundant states. For example, in a scenario where users $u_1$ and $u_2$ swap positions but exhibit identical signal power and behavior, the agent only needs to learn one of these states. The invariable versions sort observations based on positions, removing redundant symmetry.

**Obs_config options:** (Default observation wrapper config)

```python
{
    "nodes": {
        "position": True,
        "receivedSignal": False, # sum of all received legitimate signals
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

So, the **observation wrappers** to choose are:

- Fixed-size options: `FS_COW`, `FS_iCOW`
- Variable-size options: `VS_COW`, `VS_iCOW`
- Customizable through `obs_config` dictionary

_**Note:** the `i` in `iCOW` stands for invariable_
#### Reward Wrappers
- `MovementCost`: Penalizes movement to encourage quick positioning
- `LinearReward`: Uses raw network throughput
- `SquaredReward`: Uses squared throughput for steeper rewards
- `EvaluationReward`: For evaluation purposes, same as linear reward


#### Routing Functions

[Routing.py](./reference/MANET/Routing.md) holds multiple implementations of "Routing-Algorithms" that can be used for the network. Since the thesis focuses on perfect routing, these are not really routing algorithms in the traditional sense but rather "routing" as used in the MANET simulation.

##### Implementations of the MCMF Problem

There are three implementations of the Multi-Commodity Min-Cost Flow (MCMF) problem:

1. Linear Programming-Based Implementations
These two implementations follow a similar formulation:

- `multi_commodity_max_flow_pulp_1`
  Uses `PuLP` for solving the MCMF problem.

- `multi_commodity_max_flow_xpress`
  Uses `Xpress` via `PuLP`.  
  Limitation: The XPress solver struggles with handling more than 500 constraints, making it impractical for larger networks.

2. Alternative `PuLP` Implementation
- `multi_commodity_max_flow_pulp_2  `
  Another formulation using `PuLP`.

3. Gurobi Implementation
- `multi_commodity_max_flow_gurobi`
  Uses `Gurobi` via `Pyomo`. Here, you need to install Gurobi to use it. Out of all LP formulations this is the fastest in execution.

All MCMF implementations should be correct, but they tend to be rather slow in practice.

##### Faster Max Flow-Based Implementation
Two simpler but faster approaches using only a single commodity max flow algorithm, which both have their downside.
- `sequential_max_flow`:  For each (source, sink, demand) tuple, the max flow algorithm is executed sequentially with flow limited by the demand, and edge capacities are reduced accordingly.
- `artificialartifical_src_sink_max_flow`: Adding an artificial source node and an artificial sink node to then only execute a single commodity max flow algorithm.

In practice the `sequential_max_flow` algorithm was used, since it is a lot faster and not far off from the MCMF optimization algos.

### 2.6 Supervise the learning process


```python
from tensorboard import program
from Utils import training_utils as ut

# Usually agents' performances are logged in out/logs/ + current date
log_dir = 'out/logs/' + ut.getDayMonthAsString() # adapt as needed

# Set up TensorBoard application
tb = program.TensorBoard()

# Configure the arguments for TensorBoard
tb.configure(argv=['tensorboard', 'serve', '--logdir', log_dir, '--host', 'localhost'])

# Start TensorBoard
tb.main()
```

## 3. Retraining Agents

### 3.1 Setup for Retraining

Same trainer module is used


```python
import train_MANETAgent as trainer
```

### 3.2 Retraining with Configuration File

Retrain is different from training, in that there is already a training configuration stored for the previously trained model. Hence, all parameters can be loaded from any documented training configuration files (usually "01_training_config.yaml"). However, in retraining, one might want to change some parameters. 

When retraining, the only parameter required in the retraining config file is `dir_of_agent_to_load`. This will be used to load the model `.zip` file from that directory. If the `steps_of_agent_to_load` parameter isn't specified, it will simply load the RL-model `.zip` file with the most steps. Otherwise, it will use the specified model. 

It will also load the training parameters from the `"01_training_config.yaml"` file, if it exists, or otherwise from any other newer config `.yaml` file in that directory. To adjust any of those parameters, simply specify them in the config.

The parameters to adjust you can find further below at [8.2](#82-all-retraining-config-file-options) or look at `src/examples/retrain/example_retrain_config.yaml` to find an example of a configuration.


```python
trainer.retrain_agent_from_config(
    config_path="src/examples/retrain/example_retrain_config.yaml"
)
```

### 3.3 Retraining with Direct Function Call

Choose the dir of the agent to load


```python
dir_of_agent_to_load = "path/to/your/model"
```

Start retraining


```python
trainer.retrain_agent(
    dir_of_agent_to_load=dir_of_agent_to_load,
    render=False,  # Overwrite parameter from "01_training_config.yaml"
    save_freq=2_000_000,  # Overwrite parameter from "01_training_config.yaml"
)
```

## 4. Evaluating Agents

To evaluate and filter agents one can use the `evaluate` function from [evaluate_MANETAgents.py](./reference/evaluate_MANETAgents.md).

To make life easier, the function takes a list of `glob` patterns to determine the agents to evaluate by resolving the patterns to paths and looking for agents in that path. If the pattern doesn't specify a specific model .zip file the evaluator will take the model with the most steps from that path. Further, it will wrap the MANET Environmnet according to the "01_training_config.yaml" or any other config found in that path if not available. To make life even easier, the MANETEvaluator does already group the agents by their number of nodes (highly depends on the rl agent) and users (the number of jammers is a fixed parameter given from the evaluation, the number of users can be too or they will depend on each agents training config if not specified). And for each group the performance can be compared to an agent baseline model.

Results can always be reproduced given the seed parameter to be set for the MANETEnvironments. 

Import evaluator and supporting modules


```python
from evaluate_MANETAgents import evaluate
import MANET.Environments as envs
import MANET.Agents as ag
import MANET.ConditionStrategy as cs
```

Specify agent paths using glob patterns


```python
agent_patterns = [
    "path/to/*/models",
]
```

Define baseline strategies for comparison


```python
baselines_with_cs = [
    (
        ag.GreedyMaximizer,
        cs.TrafficHostedConditionStrategy(traffic_threshold=100e6),
    ),
    (
        ag.GreedyMaximizer,
        cs.ThroughputConditionStrategy(threshold_throughput=100e6),
    ),
]
```

### 4.1 Evaluation Parameters

Configure evaluation parameters


```python
eval_params = {
    "seed": 3,
    
    "n_eval_eps": 1000,
    "ep_length": 500,
    "jammer_ep_steps": 150,
    "step_jammers_start_moving": 50,
    
    "deterministic": True,
    "render_mode": None,
    
    "numbOfJammers": 1,
    "numbOfUsers": [2,21], # Uniformly randomly sample from interval [2,21) for each new episode
    
    "networkConnectedAtStart": False,
    "jammersSpawnNextToUsers": False,
    "jammers_preposition": False,
    "allow_early_ep_finish": False,
    
    "attackerModel": [
        "Static_GreedyJammers",
        "Static_ClusterJammers",
        "Static_TrafficJammers",
    ], # Uniformly randomly sample attacker model from list for each episode
    
    "routing_func": "sequential_max_flow",
    
    "obs_type": "real",
    "state_loss_rate": 0.5,
    "step_size": 2,
}
```

#### 4.1.1 Special Parameters

##### Available Baseline Agents

The values sampled and whether they are viable is determined through the condition strategies.

- `RandomAgent`: Takes random actions
- `GreedyMaximizer`: Samples in eight directions, which increase the sampled value the most and moves toward best value
- `OutAndThroughAgent`: Samples direction which improves its sampled value best. Then moves away for a predetermined length in that direction, while sampling its current state. If it sampled a viable value, during moving away, it moves to the best sampled value. Otherwise, turns around and tries to go the otherway through, again sampling for the best value while doing so and residing it the best place, the place where the sampled value is maximized.
- `SpiralAgent`: Explores in spiral pattern and then moves towards the best sampled location
- `RandomSampler`: Randomly determines three positions in its close surroundings and then, while moving to each samples which location is best. After sampling all three positions it moves to the best place sampled.

##### Condition Strategies

Determine the value to sample and whether it is viable to remain at that location.

- **Local strategies**:
  - `SINR`: Maximizes signal-to-interference-plus-noise ratio
  - `Signal`: Maximizes signal strength
  - `Interference`: Minimizes interference
  - `TrafficHosted`: Maximizes hosted traffic
- **Global strategy**:
  - `Throughput`: Maximizes network throughput

For each condition stratetgy you can specify a threshold value, for which if reached, the baseline will stop its sampling cylce - considering its current location as viable enough.

##### Observation Type and State loss rate
The RL Agent is trained centralized with all observations available. But in evaluation we want to test decentralized execution with our replicated centralized agent. Thus, in evaluation one has three options to set the parameter `obs_type`. 

- `obs_type` = "full":
    Using the agent fully centralized where full refers to that all observations are available.
- `obs_type` = "random":
    Still centrailized execution, but with random state loss. The state is calculated given the information about each component (MANET nodes and users). In this case, not all this information is assumed to be available. Rather, it is updated at each time step, but the probability of the `state_loss_rate` parameter determines, for each components information whether this is updated and thus, its recent information can be used to calculated the new observation. If it is not updated, then its old information will be used. An update succeeds with probability `1 - state_loss_rate`, following a Bernoulli distribution.
- `obs_type` = "real":
    Here, the RL Agent is actually deployed decentralized, with a copy of the agent on each MANET node. Each MANET node can only collect information from other components to calculate the observations, if they have a path to it through the network. But, updates might get lost. Similarly, an update of a component's information succeeds with `1 - state_loss_rate`, following a Bernoulli distribution. The MANET node's own information is always available.

_**Hint**_: _For the `obs_type` == "full" the `state_loss_rate` parameter has no effect. And for `obs_type` == "random" if `state_loss_rate` == 0, then no updates get lost, thus, it is equal to `obs_type`=="full"._

#### 4.1.2 Summary

| Parameter                   | Description                                                                 | Default                     |
|-----------------------------|-----------------------------------------------------------------------------|-----------------------------|
| `n_eval_eps`                | Number of episodes to evaluate over                                        | `20`                        |
| `ep_length`                 | Length of each episode                                                     | `1,000`                     |
| `jammer_ep_steps`           | Steps before jammers become active                                         | `0`                         |
| `jammers_preposition`       | Whether jammers are prepositioned before evaluation starts                 | `False`                     |
| `seed`                      | The random seed used for the environment and the RL agent                 | `42`                        |
| `deterministic`             | Whether the agent's actions are deterministic                             | `True`                      |
| `obs_type`                  | Type of observation used by the agent (e.g., "full", "partial")           | `"full"`                    |
| `state_loss_rate`           | Rate of state loss in the environment                                      | `None`                      |
| `routing_func`              | Routing function used for network communication                           | `sequential_max_flow`       |
| `numbOfJammers`             | Number of jammers in the environment                                      | `None`                      |
| `numbOfUsers`               | Number of users in the environment                                        | `None`                      |
| `render_mode`               | Rendering mode for the environment ("plot" shows a GUI, None runs simulation in background)           | `None`                      |
| `networkConnectedAtStart`   | Whether the network is connected at the start of the evaluation           | `False`                     |
| `jammersSpawnNextToUsers`   | Whether jammers spawn near users                                           | `False`                     |
| `attackerModel`             | Model used by the attackers (e.g., `Static_GreedyJammers`)                | `Static_GreedyJammers`      |
| `allow_early_ep_finish`     | Whether episodes can end early if certain conditions are met              | `False`                     |
| `step_size`                 | Step size for node movement                                               | `2`                         |
| `step_jammers_start_moving` | Step at which jammers start moving                                         | `0`                         |
| `metric_param`              | Weighting parameter for quick evaluation metrics, only important for filtering          | `0.75`                      |
| `filterval`                 | Divisor used to filter saved models (only models stored at steps divisible by this value are considered), only important for filtering

Create the evaluator instance and run the evaluation

### 4.2 Run the evaluation

Set the `eval_type` to "grouped_eval". Grouped describes, that the agents are automatically grouped by number of nodes, users and jammers, such that if a Baseline is used for comparison, a Baseline is evaluated too for each group. Further, one can use the evaluation function to filter models or agents, by evaluating which performs best, given the evaluation parameters.

#### Logging the results

When evaluating obviously we typically want to log the results of that evaluation. Altough, you can run the evaluation without logging anything an still get some results output in the console. You can set the following parameters:

- `log_dir`: directory in which your results are supposed to be stored to
- `log_eval`: if true logs the throughput and the sum of distances moved by all agents for each timestep into a `.csv` file. The `.csv` file is named after the agent or baseline, but shortened, to avoid too long directories. when it is named after an rl-agent, the folder name of where the rl agent was stored in is prepended to the `.csv` file's name.
- `log_state`: if true logs the properties of all components at the start of each episode into a `state_logs.json` file.
- `log_results`: if true allows to log either for evaluation or the state. if false nothing gets logged


```python
evaluate(
    env_constr=envs.Static_MANETEnv,
    agent_patterns=agent_patterns,
    baselines_with_cs=baselines_with_cs,
    eval_params=eval_params,
    eval_type="grouped_eval",
    log_dir="out/evaluations", # directory to log the evaluatons to
    log_results=True,
    log_eval=True,
    log_state=True,
)
```

### 4.3 Evaluate from a config file

Further, one can do the same thing but put all parameters in a config `.yaml` file. You can find an example config file `src/examples/evaluate/example_eval_config.yaml` and an example python script at `src/examples/evaluate/example_evaluate_from_config.py`. To do evaluate agents and baselines from a config file, you have two options. Either run the evaluation from a python file, specifying the config's location or run the `evaluate.py` specifying the config file's location with the command line arguments:

#### 4.3.1 Evaluation from config using a python script

Import the necessary function:


```python
from evaluate_MANETAgents import evaluate_from_config
```

Run the function, specifying the location of you configuration file:


```python
evaluate_from_config("src\examples\evaluate\example_eval_config.yaml")
```

#### 4.3.2 Evaluation running python from terminal

Run the following command and simply specify the path thereafter.


```python
python src/evaluate.py src\examples\evaluate\example_eval_config.yaml
```

## 5. Filter


```python
from evaluate_MANETAgents import evaluate
import MANET.Environments as envs
```

During training, many models of an agent are typically stored at different steps, given the `save_freq` that was set, to allow selecting the best one later. We have seen how we can evaluate agents. Using the same function one can further for each agent filter its best model by setting `eval_type="filter_models"` or for multiple agents filter the agent that performs best under the set evaluation conditions (through the `eval_params`) by setting `eval_type="filter_best_agent"`.  When filtering, no baselines will be evaluated. Otherwise, filtering behaves the same to the evaluation, in terms of `eval_params` but also you can filter agents and models using config files.

__Key Features__

- **Metric Parameter**: For both filtering models and agents we need to determine a metric to compare the models or agents performance. For each episode the mean throughput before the jammers activate $\bar{\tau}_\text{before}$ and after they do $\bar{\tau}_\text{after}$, is taken. Then, we take the mean of each over all episodes: ${\tau}_\text{before}^{\ast}$ and ${\tau}_\text{after}^{\ast}$. 
    So, the metric, given the `metric_param` we for now denote as $\alpha$ is $\tau_\text{metric} = (1 - \alpha) * {\tau}_\text{before}^{\ast} + \alpha * {\tau}_\text{after}^{\ast}$

    Use the `metric_param`, here called $\alpha$ to weight, which is more important to you. At the end of evaluation of all models or agents, the one with highest value for $\tau_\text{metric}$ is chosen.

### 5.1 Filter Best Models

So, filtering models means for one agent folder to filter the stored models `rl_model_..._steps.zip` at different steps, by setting `eval_type="filter_models"`.

__Key Features__

- **Agent Patterns**: Provide patterns for paths where agent models are stored. The function evaluates and selects the best model for each pattern based on the given evaluation parameters. Patterns are resolved to the folder that holds all the agent model files `rl_model_..._steps.zip`.

- **Filterval**: Set `filterval` to a value (e.g., `2_000_000`) to skip unnecessary evaluations. Only models with
  a step count satisfying `step_count mod filterval == 0` will be evaluated, significantly speeding up the process.

- **Save Frequency**: During training, a smaller `save_freq` results in more models being stored. This can make the filtering process longer. To address this, use the `filterval` parameter.

The filtered models are copied to a new directory automatically. Usually, if e.g. you filter the models of the agent at `"out/models/{some_date}/{name_of_agent_folder}"` the best model will be copied to `"out/models/{some_date}_filtered/{name_of_agent_folder}"`. Theoretically, you can still log results, but watch out for agents with similar names.

#### Example


```python
agent_patterns = [
    "path/to/your/models",
]

# * Set the parameters for the evaluation.
eval_params = {
    "seed": 3,
    #
    "n_eval_eps": 50,
    "ep_length": 1000,
    "jammer_ep_steps": 500,
    "step_jammers_start_moving": 100,
    #
    "deterministic": True,
    "render_mode": None,
    #
    "numbOfJammers": 1,
    "numbOfUsers": (2, 21),
    #
    "networkConnectedAtStart": False,
    "jammersSpawnNextToUsers": False,
    "jammers_preposition": False,
    "allow_early_ep_finish": False,
    #
    "attackerModel": [
        "Static_GreedyJammers",
        "Static_ClusterJammers",
        "Static_TrafficJammers",
    ],
    #
    "routing_func": "sequential_max_flow",
    #
    "obs_type": "real",
    "state_loss_rate": 0.4,
    "step_size": 2,
    #
    "metric_parm": 0.75,
    "filterval": 2_000_000
}

evaluate(
    env_constr=envs.Static_MANETEnv,
    agent_patterns=agent_patterns,
    eval_params=eval_params,
    eval_type="filter_models",
    log_dir=None,
    log_results=False,
    log_state=True,
    log_eval=False,
)
```

### 5.2 Filter Best Agents

To filter for the best agent amidst many RL agents you have previously trained, you again can use a list of glob patterns. These patterns should again either resolve the an agent's model file (`rl_model_..._steps.zip`) or the agent's folder that holds said files. However, if you only specify the agent's folder through the patterns, then for evaluation the model with the highest number of steps will be selected. The best agent's model file of each group will at the end be copied into the `log_dir` folder you can specify in the `evaluate()` method. 

#### Example


```python
# * Set the paths of the agents to load
agent_patterns = [
    "path/to/*/models",
    "*/to/your/models"
]

# * Set the parameters for the evaluation.
eval_params = {
    "seed": 3,
    #
    "n_eval_eps": 50,
    "ep_length": 1000,
    "jammer_ep_steps": 500,
    "step_jammers_start_moving": 100,
    #
    "deterministic": True,
    "render_mode": None,
    #
    "numbOfJammers": 1,
    "numbOfUsers": (2, 21),
    #
    "networkConnectedAtStart": False,
    "jammersSpawnNextToUsers": False,
    "jammers_preposition": False,
    "allow_early_ep_finish": False,
    #
    "attackerModel": [
        "Static_GreedyJammers",
        "Static_ClusterJammers",
        "Static_TrafficJammers",
    ],
    #
    "routing_func": "sequential_max_flow",
    #
    "obs_type": "real",
    "state_loss_rate": 0.4,
    "step_size": 2,
    #
    "metric_parm": 0.75,
}

evaluate(
    env_constr=envs.Static_MANETEnv,
    agent_patterns=agent_patterns,
    eval_params=eval_params,
    eval_type="filter_best_agent",
    log_dir="out/test/best_agents",
    log_results=False,
    log_state=False,
    log_eval=False,
)

```

## 6. Command-Line Training

#### Overview
`src/train.py` allows for training or retraining a reinforcement learning agent by executing a terminal command.  
It determines how to proceed based on the provided path and optional arguments.

The behavior depends on **whether the path points to a file or a directory**:
1. **If the path is a file:**
   - If it's a `.yaml` config file → Trains or retrains using that config.
   - If it's an agent `.zip` file → Retrains that specific agent using `01_training_config.yaml`.

2. **If the path is a directory:**
   - If agent files are present:
     - It checks `01_training_config.yaml`:
       - If training was **not** completed → Continues training, till original budget is reached.
       - If training was completed → Retrains with new budget
   - If no trained agent exists:
     - If a config file exists → Trains or retrains using that config.
     - Otherwise → Trains from scratch with default parameters.

### 6.1 Basic Usage


```python
python train.py path/to/file_or_folder [optional arguments]
```

### 6.2 Training Methods

#### Method 1: From Configuration File

Your config file should look something like as the file at `src/examples/train/example_train_config.yaml` or `src/examples/retrain/example_retrain_config.yaml` depending whether you want to train or retrain an agent. These are essentially the same as when running `example_train_from_config.py` or `example_retrain_from_config.py`.

However, it is important that you specify the parameter `mode` as either `train` or `retrain`:

```yaml
mode:
    train
```

The configuration must be a `.yaml` file. While a structured, nested format improves readability, a flatter structure is also possible (except for `obs_config`, which remains nested). The arguments and values match those in `src/examples/train/example_train_config_less_structured.yaml` and `src/examples/retrain/example_retrain_config_less_structured.yaml`, but `mode` must be explicitly set.


##### Specify config file directly

You can specify a config file directly, so exactly that config file will be considered. In the end this will do the same as the function `train_agent_from_config` or `retrain_agent_from_config`.


```python
python src/train.py src/examples/train/example_train_config.yaml
```

##### Or specify folder containing config file

If you specify the folder it should be a folder that doesn't contain any agent .zip files, since then it assumes that any config file inside that belongs to the training or retraining configuration. Hence, it may disregard you configuration file. Further, it will simply take the config, which has been added the last. In the end this will do the same as the function `train_agent_from_config` or `retrain_agent_from_config`.


```python
python src/train.py src/examples/train
```

#### Method 2: Retraining Existing Model

##### Specify model file directly

If you specify the path of a model .zip file, then it will retrain exactly that model, using the parameters from the "01_training_config.yaml" if available, else the latest config .yaml file in that same directory. You can overwrite any argument by giving it as a command line argument. E.g. maybe you want to use a different ep_length. In the end this will do the same as the `retrain_agent` method.


```python
python src/train.py path/to/your/model/rl_model_x_steps.zip --ep_length 11000
```

##### Or specify folder containing models

If you simply specify the folder containg the model .zip files, it will first check whether the "01_training_config.yaml" file's budget was completed and either finish that or retrain by using the full budget, both on the latest model / the one with the most steps. In the end this will do the same as the `retrain_agent` method.


```python
python src/train.py path/to/your/model --save_freq 1000000
```

### 6.3 Command-Line Arguments

The most important arguments you can find here. The full options can be found under [8.3](#83-all-command-line-arguments-options)

| Argument | Type | Description |
|----------|------|-------------|
| `path` | `str` | Path to folder/file containing model or config |
| `--specific_model_step` | `str` | Specific model step to load |
| `--model_dir` | `str` | Directory to save trained model |
| `--log_dir` | `str` | Directory to save logs |
| `--timesteps` | `int` | Total training timesteps |
| `--ep_length` | `int` | Length of each episode |
| `--save_freq` | `int` | Frequency of model saving |
| (and other training parameters) | | |

## 7. Config and batch speed up's

Now, as far as batch and executing for the ubelix cluster goes there are some batch file examples for training and retraining with a config file, as well as for filtering or evaluating MANET Agents. Usually, for batch jobs one would run training, retraining, evaluation or filtering from config files, but you can also run full python scripts, that hold all parameters. The batch job examples work with the example configs or scripts in the current location:

- `src/examples/train/batch_train.sh`
- `src/examples/retrain/batch_retrain.sh`
- `src/examples/filter/batch_filter.sh`
- `src/examples/evaluate/batch_evaluate.sh`

However, this might be very tedious, especially if desired to train multiple agents. Hence, there are some scripts that help to speed up these processes.

### 7.1 Config generation


```python
from src.Utils.batch_utils import ConfigGenerator as CG
import src.Utils.training_utils as ut
```

Use this file to generate multiple training, retraining, or evaluation configs. Basically, give the function the config as a dictionary. Now, since you want to create multiple configs, use a set or list for the config attributes of which you want to iterate. Use tuples, for the intervals or attacker model lists from which for each episode they are uniformly randomly sampled.

#### Example

You want to train for multiple users but with the same config

``` Python
train_dict = {
    "mode": "train",
    "storing_and_logging": {
        "model_dir": "out/models/",
        "log_dir": "out/logs/",
        "log_name": "example",
    },
    "env": {
        "name": "Static_MANETEnv",
        "parameters": {
            "numbOfNodes": {3, 4},
            "numbOfJammers": {1, 2},
            "numbOfUsers": (2, 21),
        },
    },
    "obs_wrapper": {
        "obs_type": ["random", "real"],
        "state_loss_rate": [0.8, 0.4],
    },
}
```

**Resulting Configuration Files:**

This would generate **8 configuration files** (2×2×2 = 8 combinations):

```python
# Config 1: train_config1.yaml
{
    "mode": "train",
    "storing_and_logging": {
        "model_dir": "out/models/",
        "log_dir": "out/logs/",
        "log_name": "example",
    },
    "env": {
        "name": "Static_MANETEnv",
        "parameters": {
            "numbOfNodes": 3,
            "numbOfJammers": 1,
            "numbOfUsers": (2, 21),
        },
    },
    "obs_wrapper": {
        "obs_type": "random",
        "state_loss_rate": 0.8,
    },
}

# Config 2: train_config2.yaml
{
    "mode": "train",
    "storing_and_logging": {
        "model_dir": "out/models/",
        "log_dir": "out/logs/",
        "log_name": "example",
    },
    "env": {
        "name": "Static_MANETEnv",
        "parameters": {
            "numbOfNodes": 3,
            "numbOfJammers": 1,
            "numbOfUsers": (2, 21),
        },
    },
    "obs_wrapper": {
        "obs_type": "random",
        "state_loss_rate": 0.4,
    },
}

# Config 3: train_config3.yaml
{
    "mode": "train",
    "storing_and_logging": {
        "model_dir": "out/models/",
        "log_dir": "out/logs/",
        "log_name": "example",
    },
    "env": {
        "name": "Static_MANETEnv",
        "parameters": {
            "numbOfNodes": 3,
            "numbOfJammers": 1,
            "numbOfUsers": (2, 21),
        },
    },
    "obs_wrapper": {
        "obs_type": "real",
        "state_loss_rate": 0.8,
    },
}

# ... and so on for all 8 combinations
```

**Key Behavior:**

- **Sets** (`{3, 4}`, `{1, 2}`) → **Cartesian product**: All combinations are generated
- **Lists** (`["random", "real"]`, `[0.8, 0.4]`) → **Simultaneous iteration**: Paired together (same length required)

__Notice__

You needn't fill out all parameters, but every parameter you don't specify in the config dictionary,
will be set to its default training value, defined in `train_MANETAgents.py`'s `train()` function.

Further, you can set arguments with multiple sets, which will then take the cartesian product. If you use multiple lists, then the lists are iterated over simultaneously. Thus, they must also be of same length.

For retraining, it is helpful to use a path finder to easily get a list of all agent folder paths to
retrain the agents from. Then you can use it exactly the same.

#### 7.1.1 Multiple training configs


```python
train_dict = train_dict = {
    "mode": "train",
    "storing_and_logging": {
        "model_dir": "out/models/",
        "log_dir": "out/logs/",
        "log_name": "uni_distr",
    },
    "model": {"RL_model_name": "PPO"},
    "env": {
        "name": {"Static_MANETEnv", "Dynamic_MANETEnv"},
        "render": False,
        "parameters": {
            "numbOfNodes": {1, 2, 3, 4, 5, 6},
            "numbOfJammers": (0, 5),
            "numbOfUsers": (2, 26),
            "seed": 33,
            "networkConnectedAtStart": [False, True],
            "jammersSpawnNextToUsers": False,
            "allow_early_ep_finish": True,
            "routing_func": "sequential_max_flow",
            "step_size": 1,
        },
    },
    "attacker": {
        "model": (
            "Static_GreedyJammers",
            "Static_ClusterJammers",
            "Static_TrafficJammers",
        ),
        "steps_till_jammer_active": [150,0],
    },
    "obs_wrapper": {
        "type": "VS_COW",
        "config": ...
    },
    "rew_wrapper": {"type": "LinearReward"},
    "training": {
        "timesteps": 80_000_000,
        "ep_length": 15_000,
        "save_freq": 1_000_000,
        "stat_size": 50,
    },
}

CG.generate_configs(base_config=train_dict, output_dir="src/ubelix_train", file_name="train_config")
```

#### 7.1.2 Multiple retraining configs using patterns

Resolving the patterns


```python
patterns = ["out/upload/tutorial_models/*"]
agent_paths = set(ut.findAgentFoldersPaths(*patterns))
```

Setting the retrain config dictionary and generating the config files


```python
retrain_dict = {
    "mode": "retrain",
    "loading_model": {
        "dir_of_agent_to_load": agent_paths,
    },
    "attacker": {"model": "Static_TrafficJammer", "steps_till_jammer_active": 150},
    "env": {
        "render": False,
        "parameters": {
            "seed": 79,
        },
    },
    "training": {"reset": True},
}

CG.generate_configs(base_config=retrain_dict, output_dir="src/ubelix_train", file_name="retrain_config")
```

### 7.2 Batch file creation


```python
from src.Utils.batch_utils import BatchFileHelper as BFH
```

#### 7.2.1 Batch file creation manually

Given our config files, we now want to create their corresponding batch files easily. Now, you can also use this to create batch files for python scripts that are enumerated.

##### Execution Setup  

Choose whether to **train/retrain from a config** (`is_config=True`) or **execute a standalone Python file** (`is_config=False`).  

All files to be executed must be placed in **`/src/ubelix_train/`**:  
- **Configs** are usually already stored there.  
- **Python scripts** should also be placed in this directory.  

A series of files or configs is assumed to be **enumerated from `1` to `count`**, where `count` is the total number of files.  

To avoid conflicts with batch script names (`.sh` files):  
- Use **`start_count`** to specify the starting index for naming batch files.  
- This ensures batch scripts are uniquely named and do not overwrite existing ones.  



##### Example 1

__Config:__

- **file_name** = `"train_config"`  
- **is_config** = `True`  
- **start_count** = `5`  
- **count** = `3`  

**Result:**  
- `python_rl_job5.sh` → trains from config: `train_config1`  
- `python_rl_job6.sh` → trains from config: `train_config2`  
- `python_rl_job7.sh` → trains from config: `train_config3`  


##### Example 2

__File:__  
- **file_name** = `"example_script"`  
- **is_config** = `False`  
- **start_count** = `3`  
- **count** = `2`  

**Result:**  
- `python_rl_job3.sh` → executes file: `example_script1.py`  
- `python_rl_job4.sh` → executes file: `example_script2.py`  
  

##### Code Example

Create batch files for enumerated scripts


```python
BFH.create_rl_job_scripts_from_params(
    count=2, start_count=1, file_name="example_filter", hours=24, is_config=False
)
```

Creat batch files for enumerated config files


```python
BFH.create_rl_job_scripts_from_params(
    count=10, start_count=1, file_name="train_config", hours=24, is_config=True
)
```

#### 7.2.2 Batch file creation automatically

This is an automated version of `BFH.create_rl_job_scripts_from_params(...)`, where one had to specify the name and number of configs or Python files to run. Here, one simply defines the location of the files to run as batch jobs and the maximum time for which the batch job should run. `BFH.create_rl_job_scripts_from_search(...)` works in an adaptive fashion, creating a batch file for all configs or scripts stored in the specified `config_dir`:

- If the files stored are `.yaml`, it recognizes those as configs, and `.py` as Python files.
- Since config files should include a key `mode` with a parameter that is either `train`, `retrain`, or `evaluate`, the function will be able to run the correct code.
- If there already exist batch scripts, it will avoid existing enumeration:
    Example:
        If `python_rl_job2.sh`, `python_rl_job3.sh`, and `python_rl_job4.sh` exist, then it will use `python_rl_job1.sh` or any number larger or equal to `python_rl_job5.sh`.


```python
# Create the job scripts
BFH.create_rl_job_scripts_from_search(hours=72)
```

##### Batch commands

To easily execute the generated batch files, print the necessary commands in your console.  

Batch files follow the naming pattern:  
**`python_rl_job{d+}.sh`**  

You can specify:  
- **`start_count`** → the starting index.  
- **`count`** → the number of consecutive batch files to execute.  

**Example:**  
If `start_count=1` and `count=3`, the output will be:  

```bash
sbatch job_scripts/python_rl_job1.sh
sbatch job_scripts/python_rl_job2.sh
sbatch job_scripts/python_rl_job3.sh
```


```python
BFH.get_batch_commands(count=10, start_count=1)
```

## 8. Full parameter list

### 8.1 All training config file options

| **Category**                  | **Parameter**                       | **Default Value**                         | **Description**                                              |
|--------------------------------|-------------------------------------|-------------------------------------------|--------------------------------------------------------------|
| **Storing and Logging**        | `model_dir`                         | "out/models"                              | Directory to save the trained model                           |
|                                | `log_dir`                           | "out/logs"                                | Directory to save logs                                        |
|                                | `log_name`                          | ""                                        | Name of the log file                                          |
| **Model**                      | `RL_model_name`                     | "PPO"                                     | Reinforcement Learning model to use                           |
| **Environment**                | `env`                               | "Static_MANETEnv"                         | Environment to train the agent in                             |
|                                | `render`                            | False                                     | Whether to render the environment during training             |
| **Environment Parameters**     | `numbOfNodes`                       | 5                                         | Number of MANET nodes                                         |
|                                | `numbOfJammers`                     | 1                                         | Number of jammers                                             |
|                                | `numbOfUsers`                       | 10                                        | Number of users                                               |
|                                | `seed`                              | 9                                         | Random seed for reproducibility                               |
|                                | `networkConnectedAtStart`           | False                                     | Whether the network is connected at the start                 |
|                                | `jammersSpawnNextToUsers`           | False                                     | Whether jammers spawn near users                              |
|                                | `allow_early_ep_finish`             | True                                      | Allow early episode termination                               |
|                                | `routing_func`                      | `sequential_max_flow`                     | Determines how the data is routed                             |
| **Attacker Model**             | `attackerModel`                     | "Static_ClusterJammers"                   | Type of attacker model                                        |
|                                | `steps_till_jammer_active`          | 0                                         | Steps before jammers become active                            |
|                                | `step_jammers_start_moving`         | 0                                         | Step at which jammers start moving                            |
| **Observation Wrapper**        | `obs_wrapper`                       | "VS_iCOW"                                 | Wrapper for observations                                      |
|                                | `obs_config`                        | None                                      | Configuration for the observation wrapper                     |
| **Reward Wrapper**             | `rew_wrapper`                       | "SquaredReward"                           | Wrapper for rewards                                           |
| **Training Parameters**        | `timesteps`                         | 30,000,000                                | Total training timesteps                                      |
|                                | `ep_length`                         | 10,000                                    | Maximum episode length                                        |
|                                | `save_freq`                         | 500,000                                   | Frequency of saving the model                                 |
|                                | `stat_size`                         | 10                                        | Size of statistics

### 8.2 All retraining config file options

| **Category**           | **Parameter**                   | **Description**                                                                 |
|------------------------|----------------------------------|---------------------------------------------------------------------------------|
| **Loading Model**      | `dir_of_agent_to_load`          | Directory of the agent model to load                                           |
|                        | `steps_of_agent_to_load`        | Number of steps of the agent to load                                           |
| **Storing and Logging**| `model_dir`                     | Directory to save the model                                                    |
|                        | `log_dir`                       | Directory to save the logs                                                     |
|                        | `log_name`                      | Name of the log file                                                           |
|                        | `model_path`                    | Full path to save the model                                                    |
|                        | `log_path`                      | Full path to save the logs                                                     |
|                        | `tb_logname`                    | Log name used for TensorBoard                                                  |
| **Environment**        | `env`                           | The MANET environment class or name                                            |
|                        | `render`                        | Whether to render the environment during training                              |
|                        | `numbOfJammers`                 | Number of jammers in the environment                                           |
|                        | `seed`                          | Random seed for reproducibility                                                |
|                        | `networkConnectedAtStart`       | Whether the network is connected at the start                                  |
|                        | `jammersSpawnNextToUsers`       | Whether jammers spawn next to users                                            |
|                        | `allow_early_ep_finish`         | Whether to allow early episode termination                                     |
|                        | `routing_func`                  | Routing function used (e.g., `sequential_max_flow`)                            |
|                        | `step_size`                     | Maximum step size of MANET nodes (scales actions)                              |
| **Attacker Model**     | `attackerModel`                 | Type of attacker model (or list of models)                                     |
|                        | `steps_till_jammer_active`      | Steps until jammers become active                                              |
|                        | `step_jammers_start_moving`     | Step at which jammers start moving to position                                 |
| **Observation Config** | `obs_config`                    | Dictionary configuring the observation wrapper                                 |
| **Reward Wrapper**     | `rew_wrapper`                   | Type of reward wrapper                                                         |
| **Training**           | `timesteps`                     | Total number of training timesteps                                             |
|                        | `ep_length`                     | Maximum episode length                                                         |
|                        | `save_freq`                     | Frequency of saving the model                                                  |
|                        | `stat_size`                     | Size of the statistics buffer                                                  |
|                        | `reset`                         | Whether to reset the model training                                            |

### 8.3 All command-line arguments options

#### Required Argument
| Argument | Type | Description |
|----------|------|-------------|
| `path` | `str` | Path to the folder or file containing the model or config. |

#### Optional Arguments
| Argument | Type | Description |
|----------|------|-------------|
| `--specific_model_step` | `str` | Specific model step to load. |
| `--model_dir` | `str` | Directory where the trained model will be saved. |
| `--log_dir` | `str` | Directory where training logs will be stored. |
| `--log_name` | `str` | Name of the log file. |
| `--RL_model_name` | `str` | Name of the reinforcement learning model. |
| `--env` | `str` | Environment name for training. |
| `--render` | `bool` | Enable environment rendering during training. |
| `--numbOfNodes` | `int` | Number of nodes in the environment. |
| `--numbOfJammers` | `Union[int, Tuple[int, int]]` | Number of jammers in the environment (single value or range). |
| `--numbOfUsers` | `Union[int, Tuple[int, int]]` | Number of users in the environment (single value or range). |
| `--seed` | `int` | Random seed for reproducibility. |
| `--networkConnectedAtStart` | `bool` | Whether the network is initially connected. |
| `--jammersSpawnNextToUsers` | `bool` | Whether jammers spawn next to users. |
| `--allow_early_ep_finish` | `bool` | Allow early termination of episodes. |
| `--step_size` | `int` | Maximum step size of MANET nodes. |
| `--attackerModel` | `Union[str, List[str]]` | Type of attacker model used. |
| `--steps_till_jammer_active` | `int` | Number of steps before jammers activate. |
| `--step_jammers_start_moving` | `int` | Step at which jammers start moving. |
| `--obs_wrapper` | `str` | Name of the observation wrapper. |
| `--obs_config` | `str` | Configuration for the observation wrapper (JSON string). |
| `--rew_wrapper` | `str` | Name of the reward wrapper. |
| `--timesteps` | `int` | Total number of training timesteps. |
| `--ep_length` | `int` | Length of each episode. |
| `--save_freq` | `int` | Frequency at which the model is saved. |
| `--stat_size` | `int` | Size of the collected statistics. |
| `--routing_func` | `Union[Callable, str]` | Function or strategy
