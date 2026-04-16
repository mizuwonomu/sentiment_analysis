import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


FIGURE_DIR = Path("reports") / "figure"


def load_config(config_path: str | os.PathLike = "configs/baseline.yaml") -> Dict[str, Any]:
    """Load YAML configuration file with PyYAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sanitize_dropout(dropout_value: float) -> str:
    return str(dropout_value).replace(".", "_")


def build_training_figure_names(model_name: str, max_features: int, dropout: float) -> Dict[str, str]:
    """Build standardized output names for training-process charts."""
    drop_txt = _sanitize_dropout(dropout)
    base = f"{model_name}_feature{max_features}_dropout{drop_txt}"
    return {
        "loss": f"{base}_loss.png",
        "accuracy": f"{base}_accuracy.png",
    }


def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    max_features: int,
    dropout: float,
    output_dir: str | os.PathLike = FIGURE_DIR,
) -> Dict[str, Path]:
    """Save training loss and accuracy curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_names = build_training_figure_names(model_name, max_features, dropout)
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)

    saved_paths: Dict[str, Path] = {}

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("train_loss", []), marker="o", label="Train Loss")
    if history.get("val_loss"):
        plt.plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    plt.title(f"{model_name.upper()} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = output_dir / file_names["loss"]
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["loss"] = loss_path

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("train_acc", []), marker="o", label="Train Accuracy")
    if history.get("val_acc"):
        plt.plot(epochs, history["val_acc"], marker="o", label="Val Accuracy")
    plt.title(f"{model_name.upper()} Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    acc_path = output_dir / file_names["accuracy"]
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["accuracy"] = acc_path

    return saved_paths


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    model_name: str,
    output_dir: str | os.PathLike = FIGURE_DIR,
) -> Path:
    """Save confusion matrix heatmap as [model_name]_confusion_matrix.png."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{model_name.upper()} Confusion Matrix")
    plt.tight_layout()

    output_path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_metrics_bar(
    metrics: Dict[str, float],
    model_name: str,
    metric_group_name: str = "metrics",
    output_dir: str | os.PathLike = FIGURE_DIR,
) -> Path:
    """Save evaluation metric bars as [model_name]_[metrics].png."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=metric_names, y=metric_values, palette="viridis")
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.title(f"{model_name.upper()} {metric_group_name.capitalize()}")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()

    output_path = output_dir / f"{model_name}_{metric_group_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path
