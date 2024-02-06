import json
import os
import subprocess

from parea.constants import PAREA_DVC_METRICS_FILE, PAREA_DVC_YAML_FILE


def save_results_to_dvc_if_init(experiment_name: str, metrics: dict):
    if not _parea_dvc_initialized(print_output=False):
        return
    write_metrics_to_dvc(metrics)
    try:
        subprocess.run(["dvc", "exp", "save", "-n", experiment_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to save results to DVC: {e}")


def write_metrics_to_dvc(metrics: dict):
    with open(PAREA_DVC_METRICS_FILE, "w") as f:
        f.write(json.dumps(metrics, indent=2))


def _parea_dvc_initialized(print_output: bool = True) -> bool:
    print_fn = print if print_output else lambda *args, **kwargs: None
    git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True, stderr=subprocess.STDOUT).strip()
    if not os.path.exists(os.path.join(git_root, ".dvc")):
        print_fn("DVC is not initialized. Please run `dvc init` to initialize DVC.")
        return False

    if not os.path.exists(os.path.join(git_root, PAREA_DVC_YAML_FILE)):
        print_fn(f"{PAREA_DVC_YAML_FILE} is not found. Creating the file.")
        with open(os.path.join(git_root, PAREA_DVC_YAML_FILE), "w") as f:
            f.write("metrics:\n  - metrics.json\n")
    if not os.path.exists(os.path.join(git_root, PAREA_DVC_METRICS_FILE)):
        print_fn(f"{PAREA_DVC_METRICS_FILE} is not found. Creating the file.")
        write_metrics_to_dvc({})
    if PAREA_DVC_METRICS_FILE not in subprocess.check_output(['git', 'ls-files', PAREA_DVC_METRICS_FILE], cwd=git_root, text=True, stderr=subprocess.STDOUT):
        print_fn(f"{PAREA_DVC_METRICS_FILE} is not committed. Please to commit the file to your git history.")
        return False
    print_fn("DVC and Parea's DVC integration are initialized.")
    return True


# Example usage:
if __name__ == "__main__":
    print(_parea_dvc_initialized())
