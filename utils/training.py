import os
import cv2 
import numpy as np
import json
def get_next_experiment_number(base_path="experiments"):
    # Get the list of existing experiment folders
    existing_experiments = [d for d in os.listdir(base_path) if d.startswith("experiment")]
    
    # Extract the number from the folder names, e.g., 'experiment_1' -> 1
    if existing_experiments:
        experiment_numbers = [int(exp.split('_')[1]) for exp in existing_experiments]
        next_number = max(experiment_numbers) + 1
    else:
        next_number = 1
    
    return next_number

def create_experiment_folder(base_path="experiments", config_data=None):
    # Ensure the base directory exists
    os.makedirs(base_path, exist_ok=True)

    # Get the next available experiment number
    experiment_number = get_next_experiment_number(base_path)

    # Create a new experiment folder name like 'experiment_1'
    experiment_folder = os.path.join(base_path, f"experiment_{experiment_number}")

    # Create subdirectories for the experiment
    subdirs = ["checkpoints", "logs", "data" , "best" , ]
    for subdir in subdirs:  
        os.makedirs(os.path.join(experiment_folder, subdir), exist_ok=True)

    # Create a config file if data is provided
    if config_data is not None:
        config_path = os.path.join(experiment_folder, "config.json")
        with open(config_path, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)

    print(f"Experiment folder created at: {experiment_folder}")
    return experiment_folder

