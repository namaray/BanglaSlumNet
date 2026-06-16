"""
Structured results logging: writes per-run JSON and appends to all_runs.csv.
Every record carries config_hash, prompt_version, git_commit, seed for provenance.
"""

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..locate_anything.prompts import PROMPT_VERSION


def _config_hash(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class ResultsRecorder:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.runs_dir = self.results_dir / "runs"
        self.tables_dir = self.results_dir / "tables"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.tables_dir / "all_runs.csv"

    def record(
        self,
        run_id: str,
        experiment: str,
        config: dict,
        metrics: Dict[str, float],
        per_region: Optional[Dict] = None,
        checkpoint: Optional[str] = None,
    ) -> Dict:
        record = {
            "run_id": run_id,
            "experiment": experiment,
            "config": config,
            "config_hash": _config_hash(config),
            "prompt_version": PROMPT_VERSION,
            "seed": config.get("seed", 1337),
            "git_commit": _git_commit(),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "hc_iou": metrics.get("hc_iou", float("nan")),
                "all_iou": metrics.get("all_iou", float("nan")),
                "precision": metrics.get("precision", float("nan")),
                "recall": metrics.get("recall", float("nan")),
                "f1": metrics.get("f1", float("nan")),
                "specificity": metrics.get("specificity", float("nan")),
                "balanced_accuracy": metrics.get("balanced_accuracy", float("nan")),
                "pred_pos_rate": metrics.get("pred_pos_rate", float("nan")),
                "target_pos_rate": metrics.get("target_pos_rate", float("nan")),
                "map50": metrics.get("map50", float("nan")),
                "fpr_control_old_dhaka": metrics.get("fpr_control_old_dhaka", float("nan")),
                "fpr_control_gulshan": metrics.get("fpr_control_gulshan", float("nan")),
                "korail_recall": metrics.get("korail_recall", float("nan")),
                "ssim": metrics.get("ssim", None),
                "psnr": metrics.get("psnr", None),
            },
            "per_region": per_region or {},
            "checkpoint": checkpoint or "",
        }

        # Write per-run JSON
        run_json = self.runs_dir / f"{run_id}.json"
        with open(str(run_json), "w") as f:
            json.dump(record, f, indent=2, default=str)

        # Append to flat CSV
        self._append_csv(record)

        return record

    def _append_csv(self, record: Dict):
        flat = {
            "run_id": record["run_id"],
            "experiment": record["experiment"],
            "config_hash": record["config_hash"],
            "prompt_version": record["prompt_version"],
            "seed": record["seed"],
            "git_commit": record["git_commit"],
            "timestamp": record["timestamp"],
            "backbone_config": record["config"].get("model", {}).get("config", ""),
            "checkpoint": record["checkpoint"],
            **{f"metric_{k}": v for k, v in record["metrics"].items()},
        }
        write_header = not self.csv_path.exists()
        with open(str(self.csv_path), "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat)
