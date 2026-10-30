"""
Use this script to generate batch files. Choose whether you train or retrain from a config (is_config=True), or whether you execute just a python file. The files to be executed must be put in the folder /src/ubelix_train. The generated configs are usually already stored there. A series of files or configs is assumed to be enumerated from 1 to count=#files. However, to avoid conflicts with the batch scripts, since they will be named the same, use the start_count variable to indicate the index from which to continue naming the batch files (.sh files).

Examples:
    - Config:
        file_name = "train_config"
        is_config = True
        start_count = 5
        count = 3

        Result:
            - python_rl_job5.sh -> to train from config: train_config1
            - python_rl_job6.sh -> to train from config: train_config2
            - python_rl_job7.sh -> to train from config: train_config3

    - File:
        file_name = "example_script"
        is_config = False
        start_count = 3
        count = 2

        Result:
            - python_rl_job3.sh -> to execute file: example_script1.py
            - python_rl_job4.sh -> to execute file: example_script2.py
"""

from src.Utils.batch_utils import BatchFileHelper as BFH

# Create the job scripts
BFH.create_rl_job_scripts_from_params(
    count=8, start_count=1, file_name="model_test", hours=72, is_config=True
)
