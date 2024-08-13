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
        "seed": {"values": list(range(1, 6))},
        "rt_neutral_sampling": {"values": ['800']},
        "rt_user_noise_std": {"values": [500]},
        "rt_frustration_std_method": {"values": ["+"]},
        "rt_baseline_std": {"values": [0, 500, 1000]},
        "rt_w_word_count": {"values": [300, 350]},
        "rt_method": {"values": ["heuristic"]},
        "rt_bins_option_num": {"values": [0, 1, 2]},
        "rt_bins_columns_rep": {"values": ["one-hot", "ordinal", "both"]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
