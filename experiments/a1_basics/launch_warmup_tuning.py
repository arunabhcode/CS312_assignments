EXPERIMENT_KEY = "warmup-tuning-v1"
WARMUP_RATES = (0.01, 0.03, 0.1, 0.3, 0.5, 0.7)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            warmup_percent=warmup_percent,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for warmup_percent in WARMUP_RATES
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs())


if __name__ == "__main__":
    main()
