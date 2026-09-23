"""Show what one row of the tokenized training data looks like as text."""

from __future__ import annotations

import modal

from data import DEFAULT_DATA_SEED, TRAIN_SPLIT_NAME, dclm_train_dataset, load_token_dataset
from modal_utils import (
    MODAL_ENVIRONMENT,
    VOLUME_MOUNTS,
    build_image,
    timestamped_modal_app_name,
)


ROW_INDEX = 0
MAX_TOKENS = 160
TOKENIZER_REPO_ID = "Ryukijano/parameter-golf-sp4096"
TOKENIZER_FILENAME = "fineweb_4096_bpe.model"
TOKENIZER_SUBFOLDER = "tokenizers"


app = modal.App("dl-alchemy-data-row-inspector")


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=10 * 60)
def inspect_data_row():
    from huggingface_hub import hf_hub_download
    import sentencepiece as spm

    dataset = load_token_dataset(
        dclm_train_dataset(),
        role=TRAIN_SPLIT_NAME,
        data_seed=DEFAULT_DATA_SEED,
    )
    row = dataset[ROW_INDEX]["input_ids"]
    token_ids = [int(token_id) for token_id in row[:MAX_TOKENS]]

    tokenizer_path = hf_hub_download(
        repo_id=TOKENIZER_REPO_ID,
        filename=TOKENIZER_FILENAME,
        subfolder=TOKENIZER_SUBFOLDER,
        repo_type="dataset",
    )
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    return {
        "dataset_path": str(dataset.path),
        "token_ids": token_ids,
        "pieces": [tokenizer.id_to_piece(token_id) for token_id in token_ids],
        "text": tokenizer.decode(token_ids),
    }


def main() -> None:
    with modal.enable_output():
        with app.run(
            name=timestamped_modal_app_name("dl-alchemy-data-row-inspector"),
            environment_name=MODAL_ENVIRONMENT,
        ):
            result = inspect_data_row.remote()

    print(f"Dataset path: {result['dataset_path']}")
    print("Token IDs:")
    print(result["token_ids"])
    print("\nSentencePiece pieces:")
    print(result["pieces"])
    print("\nDecoded text:")
    print(result["text"])


if __name__ == "__main__":
    main()
