EXPERIMENT_KEY = "batch-size-tuning-v1"
BATCH_SIZES = (16, 32, 64, 128, 256, 512)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            batch_size=batch_size,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for batch_size in BATCH_SIZES
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs())


if __name__ == "__main__":
    main()
