EXPERIMENT_KEY = "scheduler-lr-tuning-v1"
SCHEDULERS = ("linear", "cos", "wsd0.2", "constant")
LEARNING_RATES = (1e-3, 3e-3, 1e-2)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            lr_schedule=scheduler,
            learning_rate=learning_rate,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for scheduler in SCHEDULERS
        for learning_rate in LEARNING_RATES
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs())


if __name__ == "__main__":
    main()
