"""
This script demonstrates how to evaluate multiple agents in a MANET environment.
You can evaluate models with different numbers of nodes, jammers, or users.
Certain parameters for the environment are fixed and defined in `eval_params`.

Agents can be specified using patterns or full paths. If a full agent file (.zip)
is specified, that specific agent is loaded; otherwise, the latest model in the
specified path is loaded.

To compare agents effectively, baselines can be specified. RL-agents are grouped
by nodes and users, ignoring the number of jammers (fixed in `eval_params`). Each
group of agents and baselines is evaluated in the same environment, ensuring
comparability.

Baselines are heuristic models with simple strategies based on sampling measurements
and taking actions accordingly:
    - [RandomAgent]: Chooses actions randomly.
    - [GreedyMaximizer]: Samples values in all directions and moves towards the best value.
    - [OutAndThroughAgent]: Moves away from noise or towards improvement, then
    checks the opposite direction if no large improvement is found.
    - [SpiralAgent]: Moves in a spiral to sample the best position.
    - [RandomSampler]: Randomly samples positions and moves to the best one.

Condition strategies dictate what is sampled and the threshold for satisfactory values:
    - Local:
        - [SINR]: Maximizes signal-to-interference-plus-noise ratio.
        - [Signal]: Maximizes signal strength.
        - [Interference]: Minimizes interference.
        - [TrafficHosted]: Maximizes hosted traffic.
    - Global:
        - [Throughput]: Maximizes global throughput.

Environments for evaluation:
    - [Static_MANETEnv]: A static environment where episodes end if maximum throughput is reached.
    - [Less_Static_MANETEnv]: Users move slowly in clusters; sending behavior is constant.
    - [Dynamic_MANETEnv]: Inherits from Less_Static_MANETEnv with more realistic traffic
    modeling using a Markov Chain.

Evaluation parameters explanation:
    n_eval_eps (int): Number of episodes for which the evaluation is conducted. Default is 20.
    ep_length (int): Maximum number of steps per episode. Default is 1,000.
    jammer_ep_steps (int): Number of steps before jammers become active. Default is 0.
    jammers_preposition (bool): Whether jammers are prepositioned before evaluation starts. Default is False.
    seed (int): Random seed for reproducibility. Default is 42.
    deterministic (bool): Whether the agent's actions are deterministic. Default is True.
    obs_type (str): Type of observation used by the agent (e.g., "full", "partial"). Default is "full".
    state_loss_rate (Optional[float]): Rate of state loss in the environment. Default is None.
    routing_func (Callable): Routing function used for network communication. Default is MaxFlow.sequential_max_flow.
    numbOfJammers (Optional[Union[int, Tuple[int, int]]]): Number of jammers in the environment.
        If an interval is provided, the number is sampled uniformly at the start of each episode and remains constant during the episode. Default is None.
    numbOfUsers (Optional[Union[int, Tuple[int, int]]]): Number of users in the environment.
        If an interval is provided, the number is sampled uniformly at the start of each episode and remains constant during the episode. Default is None.
    render_mode (Optional[str]): Rendering mode for the environment (e.g., "human", "rgb_array"). Default is None.
    networkConnectedAtStart (bool): Whether the network is connected at the start of the evaluation. Default is False.
    jammersSpawnNextToUsers (bool): Whether jammers spawn near users. Default is False.
    attackerModel (Type[at.AttackerModel]): Model used by the attackers (e.g., Static_GreedyJammers). Default is at.Static_GreedyJammers.
    allow_early_ep_finish (bool): Whether episodes can end early if certain conditions are met. Default is False.
    step_size (int): Step size for node movement. Default is 2.
    step_jammers_start_moving (int): Step at which jammers start moving. Default is 0.
    metric_param (float): Weighting parameter for quick evaluation metrics. Default is 0.75.
    filterval (float): Divisor used to filter saved models. Only models stored at steps divisible by this value are evaluated. Default is 0.9.
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
    "path/to/your/*"
]

# * Specify the baseline agents to be compared to.
baselines_with_cs = [
    (
        ag.GreedyMaximizer,
        cs.TrafficHostedConditionStrategy(np.inf),
    ),
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
}

evaluate(
    env_constr=envs.Static_MANETEnv,
    agent_patterns=agent_patterns,
    baselines_with_cs=baselines_with_cs,
    eval_params=eval_params,
    eval_type="grouped_eval",
    log_dir="out/evaluations/test1",
    log_results=True,
    log_state=True,
    log_eval=True,
)
