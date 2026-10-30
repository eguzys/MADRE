"""
When multiple agents are trained for the same allocation, one may wonder which agent performs best.
The `evaluate` function allows filtering the best agent by setting `eval_type="filter_best_agent"`.

### Key Features:
- **Agent Selection**: Provide a pattern of agent folders or paths. If no specific model
  (e.g., `rl_model_..._step.zip`) is specified, the function resolves to the longest-trained
  agent in each folder.

- **Evaluation Parameters**: The agents are evaluated under the `eval_params` provided. If `seed != 0`,
  the agents are run through identical scenarios for consistent comparison.

- **Best Value Calculation**: The best agent is determined using the formula:
  `best_value = (1 - metric_param) * throughput_before_jammers + metric_param * throughput_after_jammers`.

- **Group-Based Comparison**: Agents are grouped based on the number of nodes, jammers, and users.
  The agent achieving the best value in its group is copied to the specified `log_dir`.

- **Node Limitation**: While RL agents can be configured for various numbers of users and jammers,
  they are limited to a fixed number of MANET nodes. The stored configuration of the agent determines
  the number of nodes.

### Metric Parameter:
Use `metric_param` to control the weight of pre-attack and post-attack throughput in the evaluation metric:
`best_value = (1 - metric_param) * throughput_before_jammers + metric_param * throughput_after_jammers`.
"""

import numpy as np

# Evaluator class
from evaluate_MANETAgents import evaluate

# Environments
import MANET.Environments as envs

# Agents
import MANET.Agents as ag

# ConditionStrategy
import MANET.ConditionStrategy as cs

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
