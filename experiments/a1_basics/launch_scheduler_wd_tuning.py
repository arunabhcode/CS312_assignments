EXPERIMENT_KEY = "scheduler-wd-tuning-v1"
SCHEDULERS = ("linear", "cos", "wsd0.2")
WEIGHT_DECAYS = (0.01, 0.1, 0.3)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            lr_schedule=scheduler,
            weight_decay=weight_decay,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for scheduler in SCHEDULERS
        for weight_decay in WEIGHT_DECAYS
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs())


if __name__ == "__main__":
    main()
