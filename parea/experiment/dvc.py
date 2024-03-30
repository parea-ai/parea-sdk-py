import os
import subprocess

from parea.constants import PAREA_DVC_DIR, PAREA_DVC_METRICS_FILE, PAREA_DVC_YAML_FILE
from parea.utils.universal_encoder import json_dumps


def is_git_repo():
    try:
        subprocess.check_output(["git", "branch"], stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False


def save_results_to_dvc_if_init(experiment_name: str, metrics: dict):
    if not parea_dvc_initialized(only_check=True):
        return
    write_metrics_to_dvc(metrics)
    try:
        subprocess.run(["dvc", "exp", "save", "-n", experiment_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to save results to DVC: {e}")


def write_metrics_to_dvc(metrics: dict):
    git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.STDOUT).strip()
    with open(os.path.join(git_root, PAREA_DVC_METRICS_FILE), "w") as f:
        f.write(json_dumps(metrics, indent=2))


def _check_has_been_committed(git_root: str, file: str) -> bool:
    output = subprocess.check_output(["git", "log", "--", file], cwd=git_root, text=True, stderr=subprocess.STDOUT)
    return output and len(output) > 0


def parea_dvc_initialized(only_check: bool) -> bool:
    print_fn = print if not only_check else lambda *args, **kwargs: None

    if not is_git_repo():
        print_fn("Git repository is not found. Please run `git init` to initialize a git repository.")
        return False

    git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.STDOUT).strip()

    # make sure DVC is initialized
    if not os.path.exists(os.path.join(git_root, ".dvc")):
        print_fn("DVC is not initialized. Please run `dvc init` to initialize DVC.")
        return False

    # make sure dvc.yaml and metrics.json exist in .parea directory
    if not os.path.exists(os.path.join(git_root, PAREA_DVC_YAML_FILE)):
        if only_check:
            return False
        else:
            print_fn(f"{PAREA_DVC_YAML_FILE} is not found. Creating the file.")
            if not os.path.exists(os.path.join(git_root, PAREA_DVC_DIR)):
                os.mkdir(os.path.join(git_root, PAREA_DVC_DIR))
            with open(os.path.join(git_root, PAREA_DVC_YAML_FILE), "w") as f:
                f.write("metrics:\n  - metrics.json\n")
            subprocess.run(["git", "add", PAREA_DVC_YAML_FILE], cwd=git_root, check=True)
    if not os.path.exists(os.path.join(git_root, PAREA_DVC_METRICS_FILE)):
        if only_check:
            return False
        else:
            print_fn(f"{PAREA_DVC_METRICS_FILE} is not found. Creating the file.")
            if not os.path.exists(os.path.join(git_root, PAREA_DVC_DIR)):
                os.mkdir(os.path.join(git_root, PAREA_DVC_DIR))
            write_metrics_to_dvc({})
            subprocess.run(["git", "add", PAREA_DVC_METRICS_FILE], cwd=git_root, check=True)

    # make sure dvc.yaml and metrics.json are committed
    dvc_yaml_file_missing = not _check_has_been_committed(git_root, PAREA_DVC_YAML_FILE)
    dvc_metrics_file_missing = not _check_has_been_committed(git_root, PAREA_DVC_METRICS_FILE)
    if dvc_metrics_file_missing:
        print_fn(f"{PAREA_DVC_METRICS_FILE} is not committed. Please to commit the file to your git history.")
    if dvc_yaml_file_missing:
        print_fn(f"{PAREA_DVC_YAML_FILE} is not committed. Please to commit the file to your git history.")
    if dvc_metrics_file_missing or dvc_yaml_file_missing:
        return False

    print_fn("Parea's DVC integration is initialized.")
    return True
