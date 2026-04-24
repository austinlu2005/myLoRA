import json
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlLogger:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)

    def log(self, record: Dict[str, Any]):
        print(record)
        if self.path is None:
            return
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
