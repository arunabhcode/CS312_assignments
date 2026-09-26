EXPERIMENT_KEY = "scheduler-warmup-tuning-v1"
SCHEDULERS = ("linear", "constant")
WARMUP_PERCENTS = (0.0, 0.01, 0.05)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            lr_schedule=scheduler,
            warmup_percent=warmup_percent,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for scheduler in SCHEDULERS
        for warmup_percent in WARMUP_PERCENTS
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs())


if __name__ == "__main__":
    main()
