import hashlib
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pformat

from utils import USER_CONFIG, extract_import_statement, get_user


# ---------------------------------------------------------------------------
# Slurm customization block
#
# Users on a new Slurm cluster should only need to edit this block. Path-like
# values default to utils.USER_CONFIG, which itself can be configured through
# utils.py or environment variables. Set a value here when this launcher should
# use something different from utils.USER_CONFIG.

LOCAL_SLURM_CONFIG = {
    "repo_dir": None,
    "slurm_log_dir": None,
    "temp_script_dir": None,
    "uv_path": None,
    "uv_project_environment": None,
    "uv_cache_dir": None,
    "slurm_exclude": None,
}

QUEUE_CONFIGS = {
    # Stanford NLP cluster defaults.
    "sphinx": {"account": "nlp", "partition": "sphinx"},
    "jag": {"account": "nlp", "partition": "jag-standard"},
    "miso": {"account": "miso", "partition": "miso,miso-lo"},
    "aal": {"account": "aal", "partition": "aal"},
    # Example for another cluster:
    # "gpu": {"account": None, "partition": "gpu", "gpu_request": "gres"},
}

# Optional shell commands inserted before `uv run`, useful on clusters that need
# `module load ...` or a site-specific environment setup command.
ENV_SETUP_COMMANDS = ""


SLURM_TEMPLATE = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
{account_directive}
{partition_directive}
{gpu_directive}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --time=72:00:00
#SBATCH --output={slurm_log_dir}/%j.out
{dependency_directive}
{exclude_directive}

set -euo pipefail
export PYTHONUNBUFFERED=1
{env_setup_commands}
export UV_PROJECT_ENVIRONMENT="{uv_project_environment}"
export UV_CACHE_DIR="{uv_cache_dir}"

cd {repo_dir}
mkdir -p {slurm_log_dir} {temp_script_dir}

echo "User: {user}"
echo "Host: $(hostname)"
echo "Running command: {function_call}"

trap 'rm -f {script_name}' EXIT
cat > {script_name} << EOL
{import_statement}

if __name__ == '__main__':
    {function_call}
EOL

{uv_path} run python {script_name} &
child_pid=$!

forward_signal() {{
    echo "Forwarding termination signal to child $child_pid"
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid"
    exit $?
}}

trap forward_signal TERM INT
wait "$child_pid"
"""


SLURM_ARRAY_TEMPLATE = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
{account_directive}
{partition_directive}
{gpu_directive}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --time=72:00:00
#SBATCH --array=0-{array_max}{array_limit}
#SBATCH --output={slurm_log_dir}/%A_%a.out
{dependency_directive}
{exclude_directive}

set -euo pipefail
export PYTHONUNBUFFERED=1
{env_setup_commands}
export UV_PROJECT_ENVIRONMENT="{uv_project_environment}"
export UV_CACHE_DIR="{uv_cache_dir}"

cd {repo_dir}
mkdir -p {slurm_log_dir} {temp_script_dir}

echo "User: {user}"
echo "Host: $(hostname)"
echo "Array task: ${{SLURM_ARRAY_TASK_ID:-0}} / {array_max}"
echo "Running command list: {num_calls} function calls"

script_name="{script_name}.${{SLURM_ARRAY_TASK_ID:-0}}.py"
trap 'rm -f "$script_name"' EXIT
cat > "$script_name" << EOL
import os

{import_statement}

FUNCTION_CALLS = {function_calls_repr}

if __name__ == '__main__':
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task_id < 0 or task_id >= len(FUNCTION_CALLS):
        raise IndexError(
            "SLURM_ARRAY_TASK_ID=%d is outside %d function calls"
            % (task_id, len(FUNCTION_CALLS))
        )
    function_call = FUNCTION_CALLS[task_id]
    print(
        "Array task %d/%d running: %s"
        % (task_id, len(FUNCTION_CALLS), function_call),
        flush=True,
    )
    eval(function_call, globals())
EOL

{uv_path} run python "$script_name" &
child_pid=$!

forward_signal() {{
    echo "Forwarding termination signal to child $child_pid"
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid"
    exit $?
}}

trap forward_signal TERM INT
wait "$child_pid"
"""


def slurm_config_value(name):
    if name in LOCAL_SLURM_CONFIG and LOCAL_SLURM_CONFIG[name] is not None:
        return str(LOCAL_SLURM_CONFIG[name])
    return str(getattr(USER_CONFIG, name))


def queue_info(queue):
    if queue not in QUEUE_CONFIGS:
        raise ValueError(
            f"Invalid queue {queue!r}. Add it to QUEUE_CONFIGS at the top of "
            "slurm_launch.py for your cluster."
        )
    queue_config = QUEUE_CONFIGS[queue]
    return (
        queue_config.get("account"),
        queue_config.get("partition"),
        queue_config.get("gpu_request", "gpus-per-task"),
    )


def account_directive(account):
    return f"#SBATCH --account={account}" if account else ""


def partition_directive(partition):
    return f"#SBATCH --partition={partition}" if partition else ""


def gpu_directive(gpus, gpu_request):
    if gpu_request == "gpus-per-task":
        return f"#SBATCH --gpus-per-task={gpus}"
    if gpu_request == "gres":
        return f"#SBATCH --gres=gpu:{gpus}"
    if not gpu_request:
        return ""
    raise ValueError(f"Invalid gpu_request {gpu_request!r}. Use 'gpus-per-task' or 'gres'.")


def _safe_name(text):
    safe_name = (
        text.replace("(", "_")
        .replace(")", "_")
        .replace(" ", "_")
        .replace(",", "_")
        .replace("'", "")
        .replace('"', "")
    )
    return "".join(c for c in safe_name if c.isalnum() or c in "_-")


def _script_name(label, hash_input):
    safe_name = _safe_name(label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    call_hash = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:8]
    prefix = f"{timestamp}_{call_hash}"
    safe_suffix = safe_name[-100 + len(prefix) :] if len(safe_name) > 100 - len(prefix) else safe_name
    return str(Path(slurm_config_value("temp_script_dir")) / f"{prefix}_{safe_suffix}.py")


def get_script_name(function_call):
    return _script_name(function_call, function_call)


def get_array_script_name(function_calls):
    return _script_name(
        f"array_{len(function_calls)}_function_calls",
        "\n".join(function_calls),
    )


def exclude_directive():
    exclude = slurm_config_value("slurm_exclude")
    if exclude:
        return f"#SBATCH --exclude={exclude}"
    return ""


def dependency_directive(dependency):
    if dependency:
        return f"#SBATCH --dependency={dependency}"
    return ""


def submit_script(script):
    result = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    return match.group(1) if match else None


def launch_job(function_call, queue, gpus, mem=None, cpus=16, dependency=None):
    script_name = get_script_name(function_call)
    import_statement = extract_import_statement()
    account, partition, gpu_request = queue_info(queue)
    script = deepcopy(SLURM_TEMPLATE).format(
        import_statement=import_statement,
        function_call=function_call,
        account_directive=account_directive(account),
        partition_directive=partition_directive(partition),
        gpu_directive=gpu_directive(gpus, gpu_request),
        gpus=gpus,
        mem=mem if mem is not None else 64 * gpus,
        cpus=cpus,
        script_name=script_name,
        slurm_log_dir=slurm_config_value("slurm_log_dir"),
        temp_script_dir=slurm_config_value("temp_script_dir"),
        uv_path=slurm_config_value("uv_path"),
        uv_project_environment=slurm_config_value("uv_project_environment"),
        uv_cache_dir=slurm_config_value("uv_cache_dir"),
        repo_dir=slurm_config_value("repo_dir"),
        env_setup_commands=ENV_SETUP_COMMANDS,
        dependency_directive=dependency_directive(dependency),
        exclude_directive=exclude_directive(),
        user=get_user(),
    )
    return submit_script(script)


def launch_job_array(
    function_calls,
    queue,
    gpus,
    mem=None,
    cpus=16,
    max_concurrent=None,
    dependency=None,
):
    function_calls = list(function_calls)
    if not function_calls:
        raise ValueError("launch_job_array requires at least one function call.")
    if max_concurrent is not None and max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive when set.")

    script_name = get_array_script_name(function_calls)
    import_statement = extract_import_statement()
    account, partition, gpu_request = queue_info(queue)
    array_limit = f"%{max_concurrent}" if max_concurrent is not None else ""
    script = deepcopy(SLURM_ARRAY_TEMPLATE).format(
        import_statement=import_statement,
        function_calls_repr=pformat(function_calls, width=120),
        account_directive=account_directive(account),
        partition_directive=partition_directive(partition),
        gpu_directive=gpu_directive(gpus, gpu_request),
        gpus=gpus,
        mem=mem if mem is not None else 64 * gpus,
        cpus=cpus,
        array_max=len(function_calls) - 1,
        array_limit=array_limit,
        num_calls=len(function_calls),
        script_name=script_name,
        slurm_log_dir=slurm_config_value("slurm_log_dir"),
        temp_script_dir=slurm_config_value("temp_script_dir"),
        uv_path=slurm_config_value("uv_path"),
        uv_project_environment=slurm_config_value("uv_project_environment"),
        uv_cache_dir=slurm_config_value("uv_cache_dir"),
        repo_dir=slurm_config_value("repo_dir"),
        env_setup_commands=ENV_SETUP_COMMANDS,
        dependency_directive=dependency_directive(dependency),
        exclude_directive=exclude_directive(),
        user=get_user(),
    )
    return submit_script(script)
