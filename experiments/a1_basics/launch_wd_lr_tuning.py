EXPERIMENT_KEY = "wd-lr-tuning-v1"
WEIGHT_DECAYS = (0.01, 0.1, 0.3)
LEARNING_RATES = (1e-3, 3e-3, 1e-2)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for weight_decay in WEIGHT_DECAYS
        for learning_rate in LEARNING_RATES
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs())


if __name__ == "__main__":
    main()
