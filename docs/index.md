# MANETNodesRelocation

## Concept

MANETNodesRelocation addresses the following problem:

**Problem:** A well-connected Mobile Ad Hoc Network (MANET) is used as a backbone for end user communication. One or more malicious attackers attempt to jam the network, disrupting communication.

**Solution:** The proposed solution involves relocating MANET nodes using a data-driven approach based on Reinforcement Learning (RL).

MANETNodesRelocation provides multiple MANET environments, ranging from static to highly dynamic. These environments simulate various scenarios and are implemented as subclasses of `GymEnv` from [Gymnasium](https://gymnasium.farama.org/#). This compatibility allows for training RL agents on these environments, such as those from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/). Additionally, the project includes scripts that simplify model training, retraining, evaluating and filtering by allowing user of this to simply set the desired parameters, while taking care of everything else.
