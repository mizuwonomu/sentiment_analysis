from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath("."))

import joblib
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import FeedbackProcessor
from src.trainer import build_model, resolve_device, set_seed
from src.utils import load_config


SEEDS = [67, 42, 36, 2026]
METRIC_KEYS = [
    "final_test_accuracy",
    "final_test_precision",
    "final_test_recall",
    "final_test_f1",
]


def load_checkpoint_state(path: Path, device: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def prepare_test_loader(config: dict[str, Any], vectorizer: Any) -> DataLoader:
    data_cfg = config["data"]
    training_cfg = config["training"]
    processor = FeedbackProcessor(max_length=data_cfg["max_length"], device="cpu")
    processor.load_data()

    test_texts = [processor.process_text(item["sentence"]) for item in processor.test_raw]
    test_labels = [item["sentiment"] for item in processor.test_raw]
    x_test = vectorizer.transform(test_texts).toarray()

    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long),
    )
    return DataLoader(test_ds, batch_size=training_cfg["batch_size"], shuffle=False)


def evaluate_seed(seed: int) -> dict[str, Any]:
    config_path = Path("outputs") / f"ann_tfidf_seed{seed}_config.yaml"
    model_path = Path("models") / f"ann_tfidf_seed{seed}_best_model.pt"
    vectorizer_path = Path("models") / f"ann_tfidf_seed{seed}_tfidf_vectorizer.pkl"

    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    device = resolve_device(config["training"]["device"])

    vectorizer = joblib.load(vectorizer_path)
    test_loader = prepare_test_loader(config, vectorizer)

    model = build_model(config).to(device)
    model.load_state_dict(load_checkpoint_state(model_path, device))
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            logits = model(x_batch.to(device))
            preds = torch.argmax(logits, dim=1)
            all_pred.extend(preds.detach().cpu().numpy())
            all_true.extend(y_batch.detach().cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true,
        all_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "final_test_accuracy": float(accuracy_score(all_true, all_pred)),
        "final_test_precision": float(precision),
        "final_test_recall": float(recall),
        "final_test_f1": float(f1),
        "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
    }


def summarize_seed_metrics(metrics_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
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


def run_test_seed_evaluation(seeds: list[int] | None = None) -> dict[str, Any]:
    seeds = SEEDS if seeds is None else seeds
    metrics_by_seed = {seed: evaluate_seed(seed) for seed in seeds}
    result = summarize_seed_metrics(metrics_by_seed)

    report_dir = Path("report")
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "ann_tfidf_test_seed_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ANN TF-IDF seed models on test split.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test_seed_evaluation(seeds=args.seeds)
