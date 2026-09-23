import torch

from train import TrainConfig, train


CONFIG = TrainConfig(
    num_train_sequences=600_000,
    run_name_suffix="local-gpu",
    wandb_online=False,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Local GPU training requires an NVIDIA GPU with "
            "a working CUDA PyTorch install."
        )
    train(CONFIG)


if __name__ == "__main__":
    main()
