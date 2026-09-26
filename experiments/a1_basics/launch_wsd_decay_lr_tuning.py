EXPERIMENT_KEY = "wsd-decay-lr-tuning-v1"
WSD_SCHEDULES = ("wsd0.1", "wsd0.2", "wsd0.3", "wsd0.5")
LEARNING_RATES = (1e-3, 3e-3)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            lr_schedule=wsd_schedule,
            learning_rate=learning_rate,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for wsd_schedule in WSD_SCHEDULES
        for learning_rate in LEARNING_RATES
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs())


if __name__ == "__main__":
    main()
