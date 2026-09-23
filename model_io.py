import json
import os
import uuid
import importlib
from pathlib import Path

import torch

from model_config import LMConfig, model_dtype_for_precision
from modeling import AutoregressiveLM


MODEL_FORMAT_VERSION = 1
TRAINING_CHECKPOINT_VERSION = 1
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.pt"
METADATA_FILENAME = "metadata.json"


def _atomic_json_write(payload, final_path):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(
        f"{final_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, final_path)


def _atomic_torch_save(payload, final_path):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(
        f"{final_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)


def model_config_path(model_path):
    return Path(model_path) / CONFIG_FILENAME


def model_weights_path(model_path):
    return Path(model_path) / WEIGHTS_FILENAME


def model_metadata_path(model_path):
    return Path(model_path) / METADATA_FILENAME


def is_native_model_dir(model_path):
    model_path = Path(model_path)
    return model_config_path(model_path).is_file() and model_weights_path(model_path).is_file()


def load_model_metadata(model_path):
    metadata_path = model_metadata_path(model_path)
    if not metadata_path.is_file():
        return {}
    with open(metadata_path) as f:
        return json.load(f)


def qk_norm_from_metadata(metadata, *, default=True):
    if not isinstance(metadata, dict):
        return default
    return bool(metadata.get("qk_norm", default))


def tie_word_embeddings_from_metadata(metadata, *, default=False):
    if not isinstance(metadata, dict):
        return default
    return bool(metadata.get("tie_word_embeddings", default))


def dropout_from_metadata(metadata, *, default=0.0):
    if not isinstance(metadata, dict):
        return default
    return float(metadata.get("dropout", default))


def resolve_model_builder(builder_path):
    if builder_path is None:
        return None
    if not isinstance(builder_path, str):
        raise TypeError(f"model_builder must be an import path string, got {builder_path!r}.")
    module_name, _, qualname = builder_path.partition(":")
    if not module_name or not qualname:
        raise ValueError(
            "model_builder must look like 'module:function_or_class', "
            f"got {builder_path!r}."
        )
    resolved = importlib.import_module(module_name)
    for part in qualname.split("."):
        resolved = getattr(resolved, part)
    if not callable(resolved):
        raise TypeError(f"model_builder {builder_path!r} did not resolve to a callable.")
    return resolved


def resolve_qk_norm(metadata, requested_qk_norm, model_path):
    saved_qk_norm = qk_norm_from_metadata(metadata)
    if requested_qk_norm is None:
        return saved_qk_norm
    requested_qk_norm = bool(requested_qk_norm)
    if isinstance(metadata, dict) and "qk_norm" in metadata and requested_qk_norm != saved_qk_norm:
        raise ValueError(
            f"qk_norm={requested_qk_norm} does not match saved model at "
            f"{model_path}: qk_norm={saved_qk_norm}."
        )
    return requested_qk_norm


def resolve_tie_word_embeddings(metadata, requested_tie_word_embeddings, model_path):
    saved_tie_word_embeddings = tie_word_embeddings_from_metadata(metadata)
    if requested_tie_word_embeddings is None:
        return saved_tie_word_embeddings
    requested_tie_word_embeddings = bool(requested_tie_word_embeddings)
    if (
        isinstance(metadata, dict)
        and "tie_word_embeddings" in metadata
        and requested_tie_word_embeddings != saved_tie_word_embeddings
    ):
        raise ValueError(
            f"tie_word_embeddings={requested_tie_word_embeddings} does not match "
            f"saved model at {model_path}: "
            f"tie_word_embeddings={saved_tie_word_embeddings}."
        )
    return requested_tie_word_embeddings


def resolve_dropout(metadata, requested_dropout, model_path):
    saved_dropout = dropout_from_metadata(metadata)
    if requested_dropout is None:
        return saved_dropout
    requested_dropout = float(requested_dropout)
    if isinstance(metadata, dict) and "dropout" in metadata and requested_dropout != saved_dropout:
        raise ValueError(
            f"dropout={requested_dropout} does not match saved model at "
            f"{model_path}: dropout={saved_dropout}."
        )
    return requested_dropout


def save_model(model, model_path, metadata=None):
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    config = model.config
    if not isinstance(config, LMConfig):
        config = LMConfig.from_dict(config)
    metadata = dict(metadata or {})
    metadata.setdefault("qk_norm", bool(getattr(model, "qk_norm", True)))
    metadata.setdefault(
        "tie_word_embeddings",
        bool(getattr(model, "tie_word_embeddings", False)),
    )
    metadata.setdefault("dropout", float(getattr(model, "dropout", 0.0)))
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_state": model.state_dict(),
    }
    _atomic_json_write(config.to_dict(), model_config_path(model_path))
    _atomic_torch_save(payload, model_weights_path(model_path))
    _atomic_json_write(metadata, model_metadata_path(model_path))


def load_model_config(model_path):
    model_path = Path(model_path)
    if model_path.is_file():
        payload = load_training_checkpoint(model_path)
        return model_config_from_training_checkpoint(payload, model_path)
    with open(model_config_path(model_path)) as f:
        return LMConfig.from_dict(json.load(f))


def load_training_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    version = payload.get("version")
    if version != TRAINING_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported training checkpoint version at {checkpoint_path}: {version}"
        )
    return payload


def model_config_from_training_checkpoint(payload, checkpoint_path):
    metadata = payload.get("metadata") or {}
    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(
            f"Training checkpoint at {checkpoint_path} does not contain model_config "
            "metadata."
        )
    return LMConfig.from_dict(model_config)


def load_model(
    model_path,
    device="cpu",
    precision="mp",
    qk_norm=None,
    tie_word_embeddings=None,
    dropout=None,
):
    model_path = Path(model_path)
    if model_path.is_file():
        payload = load_training_checkpoint(model_path)
        config = model_config_from_training_checkpoint(payload, model_path)
        state_dict = payload["model_state"]
        metadata = payload.get("metadata") or {}
    elif is_native_model_dir(model_path):
        config = load_model_config(model_path)
        payload = torch.load(
            model_weights_path(model_path),
            map_location="cpu",
            weights_only=False,
        )
        version = payload.get("format_version")
        if version != MODEL_FORMAT_VERSION:
            raise ValueError(f"Unsupported model format version: {version}")
        state_dict = payload["model_state"]
        metadata = load_model_metadata(model_path)
    else:
        raise FileNotFoundError(f"No native dl_alchemy model found at {model_path}.")

    qk_norm = resolve_qk_norm(metadata, qk_norm, model_path)
    tie_word_embeddings = resolve_tie_word_embeddings(
        metadata,
        tie_word_embeddings,
        model_path,
    )
    dropout = resolve_dropout(metadata, dropout, model_path)
    dtype = model_dtype_for_precision(precision)
    model_builder = metadata.get("model_builder")
    model_builder_kwargs = metadata.get("model_builder_kwargs") or {}
    builder = resolve_model_builder(model_builder)
    if builder is None:
        model = AutoregressiveLM(
            config,
            dtype=dtype,
            qk_norm=qk_norm,
            tie_word_embeddings=tie_word_embeddings,
            dropout=dropout,
        )
    else:
        model = builder(
            config,
            dtype=dtype,
            qk_norm=qk_norm,
            tie_word_embeddings=tie_word_embeddings,
            dropout=dropout,
            **model_builder_kwargs,
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model
