"""
Use this file to generate multiple training, retraining, or evaluation configs. Basically, give the function the config as a
dictionary. Now, since you want to create multiple configs, use a set or list for the config attributes of which you
want to iterate.

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

Notice:
    You needn't fill out all parameters, but every parameter you don't specify in the config dictionary,
    will be set to its default training value, defined in train_MANETAgents.py's train() function.

    Further, you can set arguments with multiple sets, which will then take the cartesian product. If you use multiple lists, then the lists are iterated over simultaneously. Thus, they must also be of same length.

    For retraining, it is helpful to use a path finder to easily get a list of all agent folder paths to
    retrain the agents from. Then you can use it exactly the same.
"""

from src.Utils.batch_utils import ConfigGenerator as CG
import src.Utils.training_utils as ut

# * Train from config
train_dict = {
    "mode": "train",
    "storing_and_logging": {
        "model_dir": "out/models",
        "log_dir": "out/logs",
        "log_name": "test",
    },
    "model": {"RL_model_name": "PPO"},
    "env": {
        "name": "Static_MANETEnv",
        "render": False,
        "parameters": {
            "numbOfNodes": set(range(1, 6)),  # 1 to 5
            "numbOfJammers": [1, 2],
            "numbOfUsers": 5,
            "seed": 9,
            "networkConnectedAtStart": False,
            "jammersSpawnNextToUsers": False,
            "allow_early_ep_finish": False,
        },
    },
    "attacker": {"model": "Static_TrafficJammer", "steps_till_jammer_active": 150},
    "obs_wrapper": {
        "type": {"VS_COW", "VS_iCOW"},
        "config": {
            "nodes": {
                "position": True,
                "receivedSignal": False,
                "interference": True,
                "signalPower": True,
                "trafficHosted": True,
                "inCapacities": True,
                "outCapacities": True,
                "sensedDemands": False,
                "sensedSenderDemands": True,
                "inUserCount": True,
                "outUserCount": True,
                "closestUserDir": True,
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
        },
    },
    "rew_wrapper": {"type": "SquaredReward"},
    "training": {
        "timesteps": 200_000_000,
        "ep_length": 23_000,
        "save_freq": 5_000_000,
        "stat_size": 15,
    },
}

# * Retrain from config
# Find paths using pattern
patterns = ["path/to/your/models/folder"]
agent_paths = ut.findAgentFoldersPaths(*patterns)

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

CG.generate_configs(train_dict, "src/ubelix_train", "train_config")
CG.generate_configs(retrain_dict, "src/ubelix_train", "retrain_config")
