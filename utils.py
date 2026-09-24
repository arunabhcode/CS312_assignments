import inspect
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from model_config import validate_precision


REPO_ROOT = Path(__file__).resolve().parent

# Student-facing configuration. Most students only edit these three lines.
CONFIG_MODAL_ENVIRONMENT = "cs312-arunabh"
CONFIG_WANDB_ENTITY = "vrshy-stanford"
CONFIG_WANDB_PROJECT = "assignments"

# Non-Modal users only: advanced local path overrides.
# Leave these as None to use the default local directories.
CONFIG_SCRATCH_ROOT = None
CONFIG_MODEL_DIR = None
CONFIG_DATA_DIR = None

# Instructors usually leave these alone.
CONFIG_MODAL_WANDB_SECRET = "dl-alchemy-wandb"
CONFIG_MODAL_APP_NAME = "dl_alchemy"
CONFIG_MODAL_VOLUME_NAME = None


def _present(value):
    return value is not None and value != ""


def config_value(name, *env_names, default=None):
    for env_name in env_names:
        value = os.environ.get(env_name)
        if _present(value):
            return value
    value = globals().get(name)
    if _present(value):
        return value
    return default


def config_str(name, *env_names, default=None):
    value = config_value(name, *env_names, default=default)
    return None if value is None else str(value)


def timestamped_modal_app_name(base_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{base_name}-{timestamp}"


@dataclass(frozen=True)
class UserConfig:
    wandb_entity: str
    wandb_project: str
    model_dir: str
    repo_dir: str
    slurm_log_dir: str
    temp_script_dir: str
    uv_path: str
    uv_project_environment: str
    uv_cache_dir: str
    slurm_exclude: str = ""


def get_user():
    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def running_on_modal():
    return os.environ.get("MODAL_IS_REMOTE") == "1"


def modal_user_config():
    data_dir = config_str(
        "MODAL_DATA_DIR",
        "DL_ALCHEMY_MODAL_DATA_DIR",
        default="/root/data",
    )
    repo_dir = config_str(
        "MODAL_REPO_DIR",
        "DL_ALCHEMY_MODAL_REPO_DIR",
        default="/root/assignments",
    )
    model_dir = config_str(
        "MODAL_MODEL_DIR",
        "DL_ALCHEMY_MODAL_MODEL_DIR",
        default=f"{data_dir}/ckpts",
    )
    return UserConfig(
        wandb_entity=config_str(
            "CONFIG_WANDB_ENTITY",
            "WANDB_ENTITY",
            default="",
        ),
        wandb_project=config_str(
            "CONFIG_WANDB_PROJECT",
            "WANDB_PROJECT",
            default="assignments",
        ),
        model_dir=model_dir,
        repo_dir=repo_dir,
        slurm_log_dir=f"{data_dir}/slurmjobs",
        temp_script_dir=f"{repo_dir}/temp_scripts",
        uv_path="uv",
        uv_project_environment="/root/.venv",
        uv_cache_dir=f"{data_dir}/uv",
    )


def default_local_user_config(user):
    repo_dir = str(REPO_ROOT)
    scratch_root = config_str(
        "CONFIG_SCRATCH_ROOT",
        "DL_ALCHEMY_SCRATCH_ROOT",
        default=str(Path.home() / "dl_alchemy"),
    )
    model_dir = config_str(
        "CONFIG_MODEL_DIR",
        "DL_ALCHEMY_MODEL_DIR",
        default=f"{scratch_root}/ckpts",
    )
    return UserConfig(
        wandb_entity=config_str("CONFIG_WANDB_ENTITY", "WANDB_ENTITY", default=user),
        wandb_project=config_str(
            "CONFIG_WANDB_PROJECT",
            "WANDB_PROJECT",
            default="assignments",
        ),
        model_dir=model_dir,
        repo_dir=repo_dir,
        slurm_log_dir=config_str(
            "SLURM_LOG_DIR",
            "DL_ALCHEMY_SLURM_LOG_DIR",
            default=str(REPO_ROOT / "slurmjobs"),
        ),
        temp_script_dir=config_str(
            "TEMP_SCRIPT_DIR",
            "DL_ALCHEMY_TEMP_SCRIPT_DIR",
            default=str(REPO_ROOT / "temp_scripts"),
        ),
        uv_path=config_str("UV_PATH", "DL_ALCHEMY_UV_PATH", default="uv"),
        uv_project_environment=config_str(
            "UV_PROJECT_ENVIRONMENT",
            "UV_PROJECT_ENVIRONMENT",
            default=str(REPO_ROOT / ".venv"),
        ),
        uv_cache_dir=config_str(
            "UV_CACHE_DIR",
            "UV_CACHE_DIR",
            default=str(Path.home() / ".cache" / "uv"),
        ),
    )


def get_user_config(user=None):
    if user is None and running_on_modal():
        return modal_user_config()
    user = user or get_user()
    config = default_local_user_config(user)
    return replace(
        config,
        wandb_entity=config_str(
            "CONFIG_WANDB_ENTITY",
            "WANDB_ENTITY",
            default=config.wandb_entity,
        ),
        wandb_project=config_str(
            "CONFIG_WANDB_PROJECT",
            "WANDB_PROJECT",
            default=config.wandb_project,
        ),
        model_dir=config_str(
            "CONFIG_MODEL_DIR",
            "DL_ALCHEMY_MODEL_DIR",
            default=config.model_dir,
        ),
        slurm_log_dir=config_str(
            "SLURM_LOG_DIR",
            "DL_ALCHEMY_SLURM_LOG_DIR",
            default=config.slurm_log_dir,
        ),
        temp_script_dir=config_str(
            "TEMP_SCRIPT_DIR",
            "DL_ALCHEMY_TEMP_SCRIPT_DIR",
            default=config.temp_script_dir,
        ),
        uv_path=config_str(
            "UV_PATH",
            "DL_ALCHEMY_UV_PATH",
            default=config.uv_path,
        ),
        uv_project_environment=config_str(
            "UV_PROJECT_ENVIRONMENT",
            "UV_PROJECT_ENVIRONMENT",
            default=config.uv_project_environment,
        ),
        uv_cache_dir=config_str(
            "UV_CACHE_DIR",
            "UV_CACHE_DIR",
            default=config.uv_cache_dir,
        ),
    )


USER_CONFIG = get_user_config()
WANDB_ENTITY = USER_CONFIG.wandb_entity
WANDB_PROJECT = USER_CONFIG.wandb_project
MODEL_DIR = USER_CONFIG.model_dir
DATA_DIR = config_str(
    "CONFIG_DATA_DIR",
    "DL_ALCHEMY_DATA_DIR",
    default=str(Path(MODEL_DIR).parent / "data"),
)
SLURM_LOG_DIR = USER_CONFIG.slurm_log_dir
TEMP_SCRIPT_DIR = USER_CONFIG.temp_script_dir
UV_PATH = USER_CONFIG.uv_path
UV_PROJECT_ENVIRONMENT = USER_CONFIG.uv_project_environment
UV_CACHE_DIR = USER_CONFIG.uv_cache_dir


class LazyBatches:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = (len(dataset) + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.num_batches
        if idx < 0 or idx >= self.num_batches:
            raise IndexError(idx)
        start = idx * self.batch_size
        batch_input_ids = self.dataset[start : start + self.batch_size]["input_ids"]
        return torch.as_tensor(batch_input_ids, dtype=torch.long)

    def __iter__(self):
        for idx in range(self.num_batches):
            yield self[idx]


def create_batches(dataset, batch_size):
    return LazyBatches(dataset, batch_size)


def format_token_count(token_count):
    token_count = int(token_count)

    def format_scaled(value, suffix):
        if value >= 100:
            text = f"{value:.0f}"
        elif value >= 10:
            text = f"{value:.1f}"
        else:
            text = f"{value:.2f}"
        text = text.rstrip("0").rstrip(".")
        return f"{text}{suffix}"

    for scale, suffix in [
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ]:
        if token_count >= scale:
            return format_scaled(token_count / scale, suffix)
    return str(token_count)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def device_type(device):
    if device is None:
        return "cuda"
    return torch.device(device).type


def autocast_context(precision="mp", device=None):
    if precision == "mp" and device_type(device) == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def precision_config(precision):
    validate_precision(precision)
    return {
        "precision": precision,
        "param_precision": "bf16" if precision == "bf16" else "fp32",
        "compute_precision": "fp32" if precision == "fp32" else "bf16",
        "loss_precision": "fp32",
        "optimizer_state_precision": "optimizer_default"
        if precision == "bf16"
        else "fp32",
        "uses_autocast": precision == "mp",
    }


def configure_deterministic_training(enabled):
    if not enabled:
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_import_statement():
    caller_frame = inspect.stack()[2]
    caller_file = caller_frame.filename
    path = os.path.relpath(caller_file, os.getcwd())
    if not path.endswith(".py"):
        raise ValueError(f"Expected Python caller path, got {path}")
    module_path = path[:-3].replace(os.sep, ".")
    return f"from {module_path} import *"
