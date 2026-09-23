import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import config_str


TOKEN_FILE = "tokens.bin"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"
VALID_DTYPES = {"uint16", "uint32"}
SHUFFLE_CHUNK_ROWS = 32_768

DEFAULT_DATA_REPO_ID = config_str(
    "DATA_REPO_ID",
    "DL_ALCHEMY_DATA_REPO_ID",
    default="kothasuhas/dl_alchemy_seq9p6m_context1024",
)
DEFAULT_DATA_REVISION = config_str(
    "DATA_REVISION",
    "DL_ALCHEMY_DATA_REVISION",
    default="main",
)
DEFAULT_DATASET_DIR_NAME = "dclm_9p6m_ctx1024"
DEFAULT_DATA_SEED = 42
TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "val"
CONTEXT_LENGTH = 1024
VAL_SEQUENCES = 1_000
DCLM_DATASET_TAG = "dclm"


@dataclass(frozen=True)
class TokenDatasetConfig:
    name: str
    tag: str
    context_length: int = CONTEXT_LENGTH
    sequence_limit: int | None = None
    path: str | None = None


def seeded_dataset_dir_name(data_seed):
    if data_seed is None:
        return DEFAULT_DATASET_DIR_NAME
    return f"{DEFAULT_DATASET_DIR_NAME}_ds{int(data_seed)}"


def default_dataset_root(data_seed=None):
    from utils import DATA_DIR

    return Path(DATA_DIR) / seeded_dataset_dir_name(data_seed)


def dataset_path(config, data_seed=None):
    if config.path is not None:
        return Path(config.path)
    return default_dataset_root(data_seed=data_seed) / config.name


def global_shuffle_prefix_indices(source_num_sequences, output_num_sequences, seed):
    """Rows for the prefix of a globally shuffled dataset.

    This is the shared ordering contract for instructor-built seeded caches and
    runtime ephemeral fallback. A fallback materializing ``output_num_sequences``
    rows should match taking the same prefix from a full seeded cache.
    """
    source_num_sequences = int(source_num_sequences)
    output_num_sequences = int(output_num_sequences)
    if output_num_sequences <= 0 or output_num_sequences > source_num_sequences:
        raise ValueError(
            f"Cannot draw {output_num_sequences} rows from "
            f"{source_num_sequences} source rows."
        )
    return np.random.default_rng(int(seed)).permutation(source_num_sequences)[
        :output_num_sequences
    ]


def dclm_train_dataset():
    return TokenDatasetConfig(
        name=TRAIN_SPLIT_NAME,
        tag=DCLM_DATASET_TAG,
        context_length=CONTEXT_LENGTH,
    )


def dclm_val_dataset():
    return TokenDatasetConfig(
        name=VAL_SPLIT_NAME,
        tag=DCLM_DATASET_TAG,
        context_length=CONTEXT_LENGTH,
        sequence_limit=VAL_SEQUENCES,
    )


def token_dataset_at(path, *, tag="custom", context_length=CONTEXT_LENGTH, sequence_limit=None):
    path = Path(path)
    return TokenDatasetConfig(
        name=path.name,
        tag=tag,
        context_length=context_length,
        sequence_limit=sequence_limit,
        path=str(path),
    )


def _prefix_length(indices):
    if isinstance(indices, range):
        if indices.start == 0 and indices.step == 1:
            return indices.stop
        return None
    try:
        length = len(indices)
    except TypeError:
        return None
    for expected, index in enumerate(indices):
        if index != expected:
            return None
    return length


class PreprocessedTokenDataset:
    def __init__(self, path, sequence_limit=None):
        self.path = Path(path)
        self._tempdir = None
        metadata_path = self.path / METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing {metadata_path}. Non-Modal users only: run "
                "`python download_data.py` first. Modal users should check shared "
                "data access."
            )

        self.metadata = json.loads(metadata_path.read_text())
        dtype = self.metadata["dtype"]
        if dtype not in VALID_DTYPES:
            raise ValueError(f"Unsupported preprocessed token dtype: {dtype}")

        self.data_seed = self.metadata.get("data_seed")
        self.total_sequences = int(self.metadata["num_sequences"])
        self.seq_len = int(self.metadata["seq_len"])
        self.dtype = np.dtype(dtype)

        if sequence_limit is None:
            self.num_sequences = self.total_sequences
        else:
            self.num_sequences = int(sequence_limit)
            if self.num_sequences <= 0 or self.num_sequences > self.total_sequences:
                raise ValueError(
                    f"sequence_limit={self.num_sequences} is invalid for "
                    f"dataset of length {self.total_sequences}."
                )

        token_path = self.path / self.metadata.get("token_file", TOKEN_FILE)
        if not token_path.exists():
            raise FileNotFoundError(
                f"Missing {token_path}. Non-Modal users only: run "
                "`python download_data.py` first. Modal users should check shared "
                "data access."
            )
        self.tokens = np.memmap(
            token_path,
            dtype=self.dtype,
            mode="c",
            shape=(self.total_sequences, self.seq_len),
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.num_sequences)
            if step == 1:
                return {"input_ids": self.tokens[start:stop]}
            return {"input_ids": self.tokens[range(start, stop, step)]}
        if idx < 0:
            idx += self.num_sequences
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(idx)
        return {"input_ids": self.tokens[idx]}

    def select(self, indices):
        length = _prefix_length(indices)
        if length is None:
            raise ValueError(
                "PreprocessedTokenDataset only supports exact prefix selection. "
                "Build a new cache for arbitrary subsets."
            )
        if length <= 0 or length > self.num_sequences:
            raise ValueError(
                f"Prefix length {length} is invalid for dataset of length "
                f"{self.num_sequences}."
            )
        selected = object.__new__(type(self))
        selected.__dict__ = self.__dict__.copy()
        # Keep total_sequences and the full memmap so a later shuffle(seed)
        # matches taking the prefix of a globally shuffled cached dataset.
        selected.num_sequences = length
        return selected

    def shuffle(self, seed=None):
        if seed is None or seed == self.data_seed:
            return self
        if self.data_seed is not None:
            raise ValueError(
                f"Cannot derive data_seed={seed} from a dataset already built with "
                f"data_seed={self.data_seed}. Load the base dataset instead."
            )
        return self._materialize_global_shuffle(seed)

    def _materialize_global_shuffle(self, seed):
        seed = int(seed)
        shuffle_parent = config_str(
            "EPHEMERAL_DATA_DIR",
            "DL_ALCHEMY_EPHEMERAL_DATA_DIR",
        )
        if shuffle_parent is not None:
            Path(shuffle_parent).mkdir(parents=True, exist_ok=True)
        tempdir = tempfile.TemporaryDirectory(
            prefix=f"dl_alchemy_{self.path.name}_ds{seed}_",
            dir=shuffle_parent,
        )
        temp_path = Path(tempdir.name) / TOKEN_FILE
        target_tokens = np.memmap(
            temp_path,
            dtype=self.dtype,
            mode="w+",
            shape=(self.num_sequences, self.seq_len),
        )

        print(
            f"Materializing data_seed={seed} from base dataset {self.path}: "
            f"{self.num_sequences:,} rows.",
            flush=True,
        )
        permutation = global_shuffle_prefix_indices(
            self.total_sequences,
            self.num_sequences,
            seed,
        )
        start_time = time.perf_counter()
        for start in range(0, self.num_sequences, SHUFFLE_CHUNK_ROWS):
            stop = min(start + SHUFFLE_CHUNK_ROWS, self.num_sequences)
            target_tokens[start:stop] = self.tokens[permutation[start:stop]]
            if stop == self.num_sequences or stop % 1_000_000 < SHUFFLE_CHUNK_ROWS:
                elapsed = time.perf_counter() - start_time
                print(
                    f"  materialized {stop:,}/{self.num_sequences:,} rows "
                    f"in {elapsed:.1f}s",
                    flush=True,
                )
        target_tokens.flush()

        shuffled = object.__new__(type(self))
        shuffled.__dict__ = self.__dict__.copy()
        shuffled.path = Path(tempdir.name)
        shuffled.metadata = dict(self.metadata)
        shuffled.metadata.update(
            {
                "data_seed": seed,
                "num_sequences": self.num_sequences,
                "shuffle": "ephemeral_global_row_permutation",
                "shuffle_source_dataset": str(self.path),
                "shuffle_source_data_seed": self.data_seed,
            }
        )
        shuffled.data_seed = seed
        shuffled.total_sequences = self.num_sequences
        shuffled.num_sequences = self.num_sequences
        shuffled.tokens = target_tokens
        shuffled._tempdir = tempdir
        return shuffled


def load_preprocessed_token_dataset(path, sequence_limit=None):
    return PreprocessedTokenDataset(path, sequence_limit=sequence_limit)


def load_token_dataset(config, role, data_seed=None):
    if not isinstance(config, TokenDatasetConfig):
        raise TypeError(
            f"{role}_dataset must be a TokenDatasetConfig, got "
            f"{type(config).__name__}."
        )

    path = dataset_path(config, data_seed=data_seed)
    try:
        dataset = load_preprocessed_token_dataset(
            path,
            sequence_limit=config.sequence_limit,
        )
    except FileNotFoundError as exc:
        if config.path is None and data_seed is not None:
            base_path = dataset_path(config, data_seed=None)
            try:
                dataset = load_preprocessed_token_dataset(
                    base_path,
                    sequence_limit=config.sequence_limit,
                )
            except FileNotFoundError as base_exc:
                raise FileNotFoundError(
                    f"Missing both the pre-shuffled default dataset for "
                    f"data_seed={data_seed} at {path} and the base default "
                    f"dataset at {base_path}. Data is not available in either "
                    f"form."
                ) from base_exc
            if role == TRAIN_SPLIT_NAME:
                print(
                    f"Missing pre-shuffled train_dataset for data_seed={data_seed} "
                    f"at {path}; loaded base dataset at {base_path} and will "
                    "materialize the shuffled train rows before training.",
                    flush=True,
                )
            else:
                print(
                    f"Missing pre-shuffled {role}_dataset for data_seed={data_seed} "
                    f"at {path}; using base dataset at {base_path}.",
                    flush=True,
                )
            path = base_path
        else:
            raise
    if dataset.seq_len != config.context_length:
        raise ValueError(
            f"Expected {role}_dataset context length {config.context_length}, "
            f"found {dataset.seq_len} in {path}."
        )

    print(
        f"Loaded {role}_dataset {config.tag}: {len(dataset):,} rows "
        f"({len(dataset) * dataset.seq_len:,} tokens) from {path}."
    )
    return dataset
