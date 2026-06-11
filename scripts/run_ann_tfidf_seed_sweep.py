from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath("."))

import numpy as np
import yaml

from src.trainer import train_baseline
from src.utils import load_config


ANN_TFIDF_SEEDS = [67, 42, 36, 2026]
METRIC_KEYS = [
    "final_val_accuracy",
    "final_val_precision",
    "final_val_recall",
    "final_val_f1",
    "best_val_loss",
]


def write_seed_config(config: dict[str, Any], seed: int, output_dir: Path) -> Path:
    seed_config = copy.deepcopy(config)
    seed_config.setdefault("training", {})["seed"] = seed
    seed_config.setdefault("tracking", {})["artifact_suffix"] = f"seed{seed}"
    seed_config["tracking"].setdefault("mlflow", {})["run_name"] = f"ann_tfidf_seed{seed}"

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"ann_tfidf_seed{seed}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(seed_config, f, sort_keys=False, allow_unicode=True)
    return config_path


def load_seed_metrics(seed: int, outputs_dir: Path = Path("outputs")) -> dict[str, float]:
    metrics_path = outputs_dir / f"ann_tfidf_seed{seed}_metrics.yaml"
    with metrics_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def summarize_seed_metrics(metrics_by_seed: dict[int, dict[str, float]]) -> dict[str, Any]:
    summary = {}
    for metric_key in METRIC_KEYS:
        values = np.array(
            [metrics[metric_key] for metrics in metrics_by_seed.values()],
            dtype=np.float64,
        )
        summary[metric_key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values)),
            "values": [float(value) for value in values],
        }

    return {
        "seeds": list(metrics_by_seed.keys()),
        "summary": summary,
        "metrics_by_seed": metrics_by_seed,
    }


def run_seed_sweep(
    config_path: str | Path = "configs/ann_tfidf_sota_fixedepochs.yaml",
    run_seeds: list[int] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    run_seeds = ANN_TFIDF_SEEDS if run_seeds is None else run_seeds
    seed_config_dir = Path("outputs") / "seed_sweep" / "configs"

    for seed in run_seeds:
        seed_config_path = write_seed_config(config, seed, seed_config_dir)
        train_baseline(str(seed_config_path))

    metrics_by_seed = {}
    for seed in run_seeds:
        metrics_by_seed[seed] = load_seed_metrics(seed)

    result = summarize_seed_metrics(metrics_by_seed)
    summary_path = Path("outputs") / "ann_tfidf_seed_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANN TF-IDF seed sweep.")
    parser.add_argument("--config", default="configs/ann_tfidf_sota_fixedepochs.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_seed_sweep(config_path=args.config)