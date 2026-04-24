from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)
