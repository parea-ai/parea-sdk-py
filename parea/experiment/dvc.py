import json
import os
import subprocess

from parea.constants import PAREA_DVC_DIR, PAREA_DVC_METRICS_FILE, PAREA_DVC_YAML_FILE


def save_results_to_dvc_if_init(experiment_name: str, metrics: dict):
    if not parea_dvc_initialized(print_output=False):
        return
    write_metrics_to_dvc(metrics)
    try:
        subprocess.run(["dvc", "exp", "save", "-n", experiment_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to save results to DVC: {e}")


def write_metrics_to_dvc(metrics: dict):
    git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.STDOUT).strip()
    with open(os.path.join(git_root, PAREA_DVC_METRICS_FILE), "w") as f:
        f.write(json.dumps(metrics, indent=2))


def parea_dvc_initialized(print_output: bool) -> bool:
    print_fn = print if print_output else lambda *args, **kwargs: None
    git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.STDOUT).strip()

    # make sure DVC is initialized
    if not os.path.exists(os.path.join(git_root, ".dvc")):
        print_fn("DVC is not initialized. Please run `dvc init` to initialize DVC.")
        return False

    # make sure dvc.yaml and metrics.json exist in .parea directory
    if not os.path.exists(os.path.join(git_root, PAREA_DVC_YAML_FILE)):
        print_fn(f"{PAREA_DVC_YAML_FILE} is not found. Creating the file.")
        with open(os.path.join(git_root, PAREA_DVC_YAML_FILE), "w") as f:
            f.write("metrics:\n  - metrics.json\n")
    if not os.path.exists(os.path.join(git_root, PAREA_DVC_METRICS_FILE)):
        print_fn(f"{PAREA_DVC_METRICS_FILE} is not found. Creating the file.")
        write_metrics_to_dvc({})

    # make sure dvc.yaml and metrics.json are committed
    files_in_parea_dvc = subprocess.check_output(["git", "ls-files", PAREA_DVC_DIR], cwd=git_root, text=True, stderr=subprocess.STDOUT)
    dvc_yaml_file_missing = PAREA_DVC_YAML_FILE not in files_in_parea_dvc
    dvc_metrics_file_missing = PAREA_DVC_METRICS_FILE not in files_in_parea_dvc
    if dvc_metrics_file_missing:
        print_fn(f"{PAREA_DVC_METRICS_FILE} is not committed. Please to commit the file to your git history.")
    if dvc_yaml_file_missing:
        print_fn(f"{PAREA_DVC_YAML_FILE} is not committed. Please to commit the file to your git history.")
    if dvc_metrics_file_missing or dvc_yaml_file_missing:
        return False

    print_fn("Parea's DVC integration is initialized.")
    return True
