import os

PAREA_OS_ENV_EXPERIMENT_UUID = "_PAREA_EXPERIMENT_UUID"
PAREA_DVC_DIR = ".parea"
PAREA_DVC_METRICS_FILE = str(os.path.join(PAREA_DVC_DIR, "metrics.json"))
PAREA_DVC_YAML_FILE = str(os.path.join(PAREA_DVC_DIR, "dvc.yaml"))
