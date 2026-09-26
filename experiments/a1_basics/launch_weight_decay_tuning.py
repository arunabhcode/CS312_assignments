EXPERIMENT_KEY = "weight-decay-tuning-v1"
WEIGHT_DECAYS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            weight_decay=weight_decay,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for weight_decay in WEIGHT_DECAYS
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs())


if __name__ == "__main__":
    main()
