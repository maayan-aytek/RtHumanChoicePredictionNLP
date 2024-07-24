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
        "ENV_HPT_mode": {"values": [False]},
        "seed": {"values": list(range(1, 4))},
        "rt_method": {"values": ['model']},
        "rt_model_min_samples_leaf": {"values": [20, 50, 100]},
        "rt_model_class_weight": {"values": ['balanced', 'balanced_subsample']},
        "rt_model_top_features": {"values": ['all', '20']},
        "rt_bins_option_num": {"values": [0]},
        "rt_bins_columns_rep": {"values": ["one-hot"]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
