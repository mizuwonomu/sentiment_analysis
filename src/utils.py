import os
from pathlib import Path
from typing import Any, Dict
import yaml


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


