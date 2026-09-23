from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import modal

from metric_logging import importable_metric_loggers
from modal_utils import (
    DEFAULT_WANDB_SECRET_NAME,
    MODAL_DATA_DIR,
    MODAL_ENVIRONMENT,
    MODAL_MODEL_DIR,
    MODAL_SHARED_DATASETS_DIR,
    MODAL_USER_DATASETS_DIR,
    VOLUME_MOUNTS,
    app,
    build_image,
    secrets,
    timestamped_modal_app_name,
    user_volume,
)

if TYPE_CHECKING:
    from train import TrainConfig


DEFAULT_MODAL_GPU = "H100"


def _print_launched_training_jobs(jobs):
    launched_jobs = [job for job in jobs if not job.get("skipped")]
    if launched_jobs:
        first_job = launched_jobs[0]
        print(
            f"Launched {len(launched_jobs)} Modal training job(s) "
            f"app={first_job['modal_app_name']!r} "
            f"environment={first_job['environment']!r} gpu={first_job['gpu']!r} "
            f"detached={first_job['detached']} "
            f"max_parallel_runs={first_job['max_parallel_runs']}."
        )
    else:
        print("No Modal training jobs launched; all requested models already exist.")
    for job in jobs:
        if job.get("skipped"):
            print(
                f"  job {job['job_index']}: skipped existing model "
                f"{job['run_name']!r}"
            )
            continue
        suffix = job["run_name_suffix"]
        if suffix is None:
            suffix = "(none)"
        print(
            f"  job {job['job_index']}: call_id={job['modal_call_id']} "
            f"run_name_suffix={suffix!r}"
        )


def _model_volume_path(run_name: str) -> str:
    relative_model_dir = PurePosixPath(MODAL_MODEL_DIR).relative_to(
        PurePosixPath(MODAL_DATA_DIR)
    )
    return str(PurePosixPath("/") / relative_model_dir / run_name)


def _completed_model_exists(config: "TrainConfig") -> tuple[bool, str | None]:
    if config.force_run or not config.save_model:
        return False, None

    from model_io import CONFIG_FILENAME, WEIGHTS_FILENAME
    from train import checked_train_config, training_run_name

    try:
        config = checked_train_config(config)
        run_name = training_run_name(config)
        entries = user_volume.listdir(_model_volume_path(run_name))
    except Exception as exc:
        print(f"Could not preflight completed model check locally: {exc}")
        return False, None

    filenames = {PurePosixPath(entry.path).name for entry in entries}
    return {CONFIG_FILENAME, WEIGHTS_FILENAME}.issubset(filenames), run_name


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    secrets=secrets(),
    gpu=DEFAULT_MODAL_GPU,
    max_containers=1,
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    timeout=24 * 60 * 60,
)
def _run_training(
    config: "TrainConfig",
    requested_gpu: str,
    requested_environment: str,
):
    from train import TrainConfig, train

    if not isinstance(config, TrainConfig):
        raise TypeError(f"config must be a TrainConfig, got {type(config).__name__}.")

    config = replace(config, model_dir=str(MODAL_MODEL_DIR))
    if config.wandb_online and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "This Modal TrainConfig has wandb_online=True, but WANDB_API_KEY is "
            "not set in the container. Create a Modal secret containing "
            f"WANDB_API_KEY named `{DEFAULT_WANDB_SECRET_NAME}`. For example: "
            f"`uv run modal secret create --force {DEFAULT_WANDB_SECRET_NAME} "
            "WANDB_API_KEY=YOUR_WANDB_API_KEY`."
        )
    try:
        start_time = time.perf_counter()
        train(config)
        elapsed_seconds = time.perf_counter() - start_time
        return {
            "gpu": requested_gpu,
            "environment": requested_environment,
            "elapsed_seconds": elapsed_seconds,
            "default_data_dir": str(MODAL_SHARED_DATASETS_DIR),
            "user_data_dir": str(MODAL_USER_DATASETS_DIR),
            "model_dir": str(MODAL_MODEL_DIR),
        }
    finally:
        user_volume.commit()


def launch_training_jobs(
    configs: list["TrainConfig"],
    *,
    gpu: str = DEFAULT_MODAL_GPU,
    environment_name: str = MODAL_ENVIRONMENT,
    max_parallel_runs: int | None = None,
):
    assert isinstance(
        configs, list
    ), "launch_training_jobs expects a list[TrainConfig]; use [config] for one job."
    if not configs:
        raise ValueError("launch_training_jobs requires at least one TrainConfig.")
    if max_parallel_runs is None:
        max_parallel_runs = len(configs)
    if max_parallel_runs < 1:
        raise ValueError(
            f"max_parallel_runs must be positive, got {max_parallel_runs}."
        )

    jobs = []
    pending_configs = []
    for job_index, config in enumerate(configs):
        exists, run_name = _completed_model_exists(config)
        if exists:
            jobs.append(
                {
                    "job_index": job_index,
                    "modal_call_id": None,
                    "modal_app_name": None,
                    "run_name": run_name,
                    "run_name_suffix": config.run_name_suffix,
                    "gpu": gpu,
                    "environment": environment_name,
                    "detached": False,
                    "max_parallel_runs": max_parallel_runs,
                    "skipped": True,
                }
            )
        else:
            pending_configs.append((job_index, config))

    if not pending_configs:
        _print_launched_training_jobs(jobs)
        return jobs

    modal_app_name = timestamped_modal_app_name()
    with modal.enable_output():
        with app.run(
            name=modal_app_name,
            detach=True,
            environment_name=environment_name,
        ):
            remote_training = _run_training.with_options(
                gpu=gpu,
                env={
                    "DL_ALCHEMY_MODAL_ENVIRONMENT": environment_name,
                    "MODAL_ENVIRONMENT": environment_name,
                },
                secrets=secrets(
                    include_wandb=any(
                        config.wandb_online for _, config in pending_configs
                    )
                ),
                max_containers=max_parallel_runs,
            )

            for job_index, config in pending_configs:
                remote_config = replace(
                    config,
                    metric_loggers=importable_metric_loggers(config.metric_loggers),
                )
                call = remote_training.spawn(
                    remote_config,
                    requested_gpu=gpu,
                    requested_environment=environment_name,
                )
                jobs.append(
                    {
                        "job_index": job_index,
                        "modal_call_id": call.object_id,
                        "modal_app_name": modal_app_name,
                        "run_name_suffix": config.run_name_suffix,
                        "gpu": gpu,
                        "environment": environment_name,
                        "detached": True,
                        "max_parallel_runs": max_parallel_runs,
                    }
                )
            jobs.sort(key=lambda job: job["job_index"])
            _print_launched_training_jobs(jobs)
            return jobs
