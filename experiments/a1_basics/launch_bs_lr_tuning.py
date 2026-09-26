EXPERIMENT_KEY = "bs-lr-tuning-v1"
BATCH_SIZES = (32, 64, 128)
LEARNING_RATES = (1e-3, 3e-3, 1e-2)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for batch_size in BATCH_SIZES
        for learning_rate in LEARNING_RATES
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs())


if __name__ == "__main__":
    main()
