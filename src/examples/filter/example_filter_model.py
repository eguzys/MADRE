"""
During training, many models of an agent are typically stored at different steps to allow selecting the best one later.
The evaluation function facilitates this by setting `eval_type="filter_models"`.

Key Features:
- **Agent Filtering**: Provide patterns for paths where agent models are stored. The function evaluates and selects
  the best model for each pattern based on the given evaluation parameters.

- **Metric Parameter**: Use `metric_param` to control the weight of pre-attack and post-attack throughput in the
  evaluation metric:
  `best_value = (1 - metric_param) * throughput_before_jammers + metric_param * throughput_after_jammers`.

- **Save Frequency**: During training, a smaller `save_freq` results in more models being stored. This can make
  the filtering process longer. To address this, use the `filterval` parameter.

- **Filterval**: Set `filterval` to a value (e.g., `2_000_000`) to skip unnecessary evaluations. Only models with
  a step count satisfying `step_count % filterval == 0` will be evaluated, significantly speeding up the process.
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
