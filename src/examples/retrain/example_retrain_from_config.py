"""
This script allows retraining an agent using a configuration file.

### Overview:
The configuration file specifies the folder containing the agent's model files to be retrained.
The retraining process loads all parameters from the agent's existing configuration files,
but these can be overwritten by specifying them in the retrain configuration file.

### Key Details:
1. **Default Configuration**:
   - By default, the script loads the "01_training_config.yaml" file from the agent's folder.
   - If no specific agent file is specified in the retrain configuration, the script will
     automatically load the latest model (i.e., the one with the most training steps).

2. **Custom Configuration**:
   - You can override default parameters by specifying them in the retrain configuration file.
   - Refer to the example configuration file:
     [example_retrain_config.yaml](src/examples/retrain/example_retrain_config.yaml).

### Usage:
- Update the `config_path` parameter to point to your retrain configuration file.
- Ensure the retrain configuration file specifies the correct folder for the agent's model files or to the actual model file itself.

"""

# * Import the train lib
import train_MANETAgent as trainer

# Start retraining
trainer.retrain_agent_from_config(config_path="src/examples/retrain/example_retrain_config.yaml")
