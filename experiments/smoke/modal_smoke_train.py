from modal_train import launch_training_jobs
from train import TrainConfig


TRAIN_CONFIG = TrainConfig(
    run_name_suffix="modal",
    force_run=True,
)


def main() -> None:
    launch_training_jobs([TRAIN_CONFIG])


if __name__ == "__main__":
    main()
