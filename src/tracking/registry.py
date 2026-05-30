"""
Experiment run registry: maps config_hash → run status.
Resume-safe: on notebook restart, the orchestration cell queries this and
only runs missing rows — protecting CUs against Colab disconnects.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


class RunRegistry:
    def __init__(self, registry_path: str):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(str(self.path)) as f:
                self._data = json.load(f)

    def _save(self):
        with open(str(self.path), "w") as f:
            json.dump(self._data, f, indent=2)

    def register(self, run_id: str, config_hash: str, experiment: str):
        if run_id not in self._data:
            self._data[run_id] = {
                "run_id": run_id,
                "config_hash": config_hash,
                "experiment": experiment,
                "status": STATUS_PENDING,
            }
            self._save()

    def set_status(self, run_id: str, status: str, result_path: Optional[str] = None):
        if run_id not in self._data:
            raise KeyError(f"run_id {run_id} not registered. Call register() first.")
        self._data[run_id]["status"] = status
        if result_path:
            self._data[run_id]["result_path"] = result_path
        self._save()

    def is_done(self, run_id: str) -> bool:
        return self._data.get(run_id, {}).get("status") == STATUS_DONE

    def pending_runs(self):
        return [v for v in self._data.values() if v["status"] in (STATUS_PENDING, STATUS_RUNNING)]

    def summary(self) -> str:
        counts = {s: 0 for s in [STATUS_PENDING, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED]}
        for v in self._data.values():
            counts[v.get("status", STATUS_PENDING)] += 1
        return " | ".join(f"{k}: {v}" for k, v in counts.items())

    def should_run(self, run_id: str, force: bool = False) -> bool:
        if force:
            return True
        status = self._data.get(run_id, {}).get("status", STATUS_PENDING)
        return status not in (STATUS_DONE,)
