# Train a MANET Agent
The `train_MANETAgent.py` file provides functionality to train and retrain MANET agents in various environments. It offers flexibility in specifying agent models, environment configurations, and training parameters for comprehensive performance optimization.

__Features__

1. **Training Agents**:
    - The `train_agent` function allows users to train MANET agents with specified configurations.
    - Users can define various parameters such as the number of nodes, jammers, users, and the type of reinforcement learning model to use.

2. **Retraining Agents**:
    - The `retrain_agent` function enables users to retrain existing MANET agents with new configurations.
    - This function supports loading pre-trained models and continuing training with updated parameters.

3. **Environment Wrapping**:
    - The file includes functionality to create and wrap environments with observation and reward wrappers.
    - This ensures that the environment is configured correctly for training or evaluation.

4. **Logging and Saving**:
    - The file supports logging training progress and saving models at specified intervals.
    - Users can specify directories for storing logs and models, ensuring organized and accessible training data.

::: train_MANETAgent
    options:
      show_root_heading: false  # Avoid showing the module name as a header
