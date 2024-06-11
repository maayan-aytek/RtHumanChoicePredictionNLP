import wandb
YOUR_WANDB_USERNAME = "maayan-aytek1"
project = "206713612"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "LSTM: SimFactor=0/4 for any features representation",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        # "ENV_HPT_mode": {"values": [False]},
        # "architecture": {"values": ["LSTM"]},
        "ENV_HPT_mode": {"values": [True]},
        "seed": {"values": list(range(1, 4))},
        "rt_method": {"values": ['model']},
        "rt_model_min_samples_leaf": {"values": [20, 50, 100]},
        "rt_model_class_weight": {"values": ['balanced', 'balanced_subsample']},
        "rt_model_top_features": {"values": ['all', '20']},
        # "rt_model_file_name": {"values": ["updated_large_model_with_strategies_7.pkl", "rf_basic_classification_model_1.pkl"]}
        # "rt_method": {"values": ['baseline']}
        # "rt_method": {"values": ['random']},
        # "rt_sampling_distribution": {"values": ["normal", "uniform"]}


        # "online_simulation_factor": {"values": [0, 4]},
        # "features": {"values": ["EFs", "GPT4", "BERT"]},
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
