"""
### Overview:
This script retrains an agent model by specifying its location. If only the directory is provided,
it will retrain the latest model (i.e., the one with the most training steps).

### Key Details:
1. **Default Behavior**:
   - By default, the script uses the parameters from the "01_training_config.yaml" file if it exists
     in the agent's folder. Otherwise, it uses the most recent configuration file in that folder.

2. **Custom Parameters**:
   - To overwrite any training parameter, simply pass it as an argument to the `retrain_agent` function.

### Usage:
- Specify the directory of the agent to retrain in the `dir_of_agent_to_load` variable.
- Call the `retrain_agent` function with the desired arguments (e.g., `render`, `save_freq`).
"""

# * Import the train lib
import train_MANETAgent as trainer

# Choose the dir of the agent to load
dir_of_agent_to_load = "path/to/your/models/folder"

# Start retraining
trainer.retrain_agent(
    dir_of_agent_to_load=dir_of_agent_to_load,
    steps_of_agent_to_load=None, # If specified, tries to retrain model with this amount of steps, other wise takes model with most steps.
    render=False,
    save_freq=2_000_000,
)
