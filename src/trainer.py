import random
import os
import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec

from .data_loader import FeedbackProcessor
from .embeddings import (
    get_word_vector,
    texts_to_mean_vectors,
    texts_to_tfidf_weighted_vectors,
    train_word2vec,
)
from .models import ANNClassifier, LSTMClassifier
from .utils import load_config

MLFLOW_AVAILABLE = importlib.util.find_spec("mlflow") is not None
if MLFLOW_AVAILABLE:
    import mlflow
    import mlflow.pytorch
else:
    mlflow = None

def flatten_dict(data: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, str]:
    """Flatten nested config for MLflow param logging."""
    items = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def _get_feature_type(config: Dict) -> str:
    return config.get("features", {}).get("type", "tfidf").lower()


def _get_feature_tag(feature_type: str) -> str:
    return "w2vec" if feature_type == "word2vec" else feature_type


def _get_artifact_name(config: Dict) -> str:
    model_name = config["models"]["model_name"].lower()
    feature_type = _get_feature_type(config)
    return f"{model_name}_{_get_feature_tag(feature_type)}"


def _is_meaningful_loss_improvement(current_loss: float, best_loss: float, min_delta: float) -> bool:
    return current_loss < best_loss - min_delta


def _is_loss_plateauing(val_losses: list[float], window: int, fluctuation_delta: float) -> bool:
    if window <= 1 or len(val_losses) < window:
        return False

    recent_losses = val_losses[-window:]
    loss_range = max(recent_losses) - min(recent_losses)
    return loss_range <= fluctuation_delta and recent_losses[-1] >= recent_losses[0]


def _prepare_feature_matrices(
    config: Dict,
    train_texts: list[str],
    val_texts: list[str],
    feature_encoder: Any = None,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    data_cfg = config["data"]
    features_cfg = config.get("features", {})
    feature_type = _get_feature_type(config)

    if feature_type == "tfidf":
        tfidf_cfg = features_cfg.get("tfidf", {})
        max_features = tfidf_cfg.get(
            "max_features",
            data_cfg.get("max_features", data_cfg.get("max_Features")),
        )
        tfidf = feature_encoder or TfidfVectorizer(max_features=max_features)
        if feature_encoder is None:
            x_train = tfidf.fit_transform(train_texts).toarray()
        else:
            x_train = tfidf.transform(train_texts).toarray()
        x_val = tfidf.transform(val_texts).toarray()
        return x_train, x_val, tfidf

    if feature_type == "word2vec":
        word2vec_cfg = features_cfg["word2vec"]
        pooling = word2vec_cfg.get("pooling", "mean")
        if pooling not in {"mean", "tfidf_weighted_mean"}:
            raise ValueError(
                "Only mean and tfidf_weighted_mean pooling are currently supported "
                "for Word2Vec features."
            )

        vector_size = word2vec_cfg["vector_size"]
        embedding_source = word2vec_cfg.get("embedding_source", "input")
        train_tokens = [text.split() for text in train_texts]
        val_tokens = [text.split() for text in val_texts]

        if isinstance(feature_encoder, dict):
            word2vec = feature_encoder["word2vec"]
            tfidf = feature_encoder.get("tfidf")
        else:
            word2vec = feature_encoder or train_word2vec(train_tokens, word2vec_cfg)
            tfidf = None

        if pooling == "mean":
            x_train = texts_to_mean_vectors(
                train_tokens,
                word2vec,
                vector_size,
                embedding_source=embedding_source,
            )
            x_val = texts_to_mean_vectors(
                val_tokens,
                word2vec,
                vector_size,
                embedding_source=embedding_source,
            )
            return x_train, x_val, {"word2vec": word2vec}

        tfidf_cfg = features_cfg.get("tfidf", {})
        max_features = tfidf_cfg.get(
            "max_features",
            data_cfg.get("max_features", data_cfg.get("max_Features")),
        )
        if tfidf is None:
            tfidf = TfidfVectorizer(max_features=max_features)
            train_tfidf = tfidf.fit_transform(train_texts)
        else:
            train_tfidf = tfidf.transform(train_texts)
        val_tfidf = tfidf.transform(val_texts)

        x_train = texts_to_tfidf_weighted_vectors(
            train_tokens,
            train_tfidf,
            tfidf.vocabulary_,
            word2vec,
            vector_size,
            embedding_source=embedding_source,
        )
        x_val = texts_to_tfidf_weighted_vectors(
            val_tokens,
            val_tfidf,
            tfidf.vocabulary_,
            word2vec,
            vector_size,
            embedding_source=embedding_source,
        )
        return x_train, x_val, {"word2vec": word2vec, "tfidf": tfidf}

    raise ValueError(f"Unsupported feature type: {feature_type}")


def prepare_ann_dataloaders(config: Dict, feature_encoder: Any = None) -> Tuple[DataLoader, DataLoader, Any]:
    data_cfg = config["data"]
    training_cfg = config["training"]
    device = resolve_device(training_cfg["device"])

    processor = FeedbackProcessor(max_length=data_cfg["max_length"], device=device)
    processor.load_data()

    train_texts = [processor.process_text(item["sentence"]) for item in processor.train_raw]
    train_labels = [item["sentiment"] for item in processor.train_raw]

    val_texts = [processor.process_text(item["sentence"]) for item in processor.val_raw]
    val_labels = [item["sentiment"] for item in processor.val_raw]

    x_train, x_val, fitted_feature_encoder = _prepare_feature_matrices(
        config,
        train_texts,
        val_texts,
        feature_encoder=feature_encoder,
    )

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(train_labels, dtype=torch.long)
    full_train_dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(full_train_dataset, batch_size=training_cfg["batch_size"], shuffle=True)

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(val_labels, dtype=torch.long)
    val_ds = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"], shuffle=False)

    return train_loader, val_loader, fitted_feature_encoder


def _load_saved_word2vec(config: Dict) -> Word2Vec:
    word2vec_cfg = config.get("features", {}).get("word2vec", {})
    model_path = word2vec_cfg.get("model_path", "models/ann_w2vec_word2vec_model.model")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Word2Vec model not found at {model_path}. "
            "Train the ANN Word2Vec baseline first or update features.word2vec.model_path."
        )
    return Word2Vec.load(str(model_path))


def _build_embedding_matrix(
    vocab: Dict[str, int],
    word2vec: Word2Vec,
    embedding_dim: int,
    embedding_source: str = "input",
) -> torch.Tensor:
    if word2vec.wv.vectors.shape[1] != embedding_dim:
        raise ValueError(
            f"LSTM embedding_dim={embedding_dim} does not match "
            f"Word2Vec vector_size={word2vec.wv.vectors.shape[1]}."
        )

    embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    for token, token_id in vocab.items():
        if token in word2vec.wv.key_to_index:
            embedding_matrix[token_id] = get_word_vector(
                token,
                word2vec,
                embedding_source=embedding_source,
            )

    return torch.tensor(embedding_matrix, dtype=torch.float32)


def prepare_lstm_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    data_cfg = config["data"]
    training_cfg = config["training"]
    processor = FeedbackProcessor(max_length=data_cfg["max_length"], device="cpu")
    processor.load_data()
    processor.build_vocab(min_freq=data_cfg.get("min_freq", 1))

    x_train, y_train, train_lengths = processor.prepare_tensors(
        processor.train_raw,
        return_lengths=True,
    )
    x_val, y_val, val_lengths = processor.prepare_tensors(
        processor.val_raw,
        return_lengths=True,
    )

    train_ds = TensorDataset(x_train.cpu(), train_lengths.cpu(), y_train.cpu())
    val_ds = TensorDataset(x_val.cpu(), val_lengths.cpu(), y_val.cpu())
    train_loader = DataLoader(train_ds, batch_size=training_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"], shuffle=False)

    word2vec = _load_saved_word2vec(config)
    word2vec_cfg = config["features"]["word2vec"]
    lstm_cfg = config["models"]["lstm"]
    embedding_dim = lstm_cfg.get("embedding_dim", word2vec_cfg["vector_size"])
    embedding_matrix = _build_embedding_matrix(
        processor.vocab,
        word2vec,
        embedding_dim,
        embedding_source=word2vec_cfg.get("embedding_source", "input"),
    )

    return train_loader, val_loader, {
        "processor": processor,
        "word2vec": word2vec,
        "embedding_matrix": embedding_matrix,
        "word2vec_path": word2vec_cfg.get("model_path", "models/ann_w2vec_word2vec_model.model"),
    }


def prepare_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, Any]:
    model_name = config["models"]["model_name"].lower()
    if model_name == "ann":
        return prepare_ann_dataloaders(config)
    if model_name == "lstm":
        return prepare_lstm_dataloaders(config)
    raise ValueError(f"Unsupported model_name: {model_name}. Supported models: ann, lstm.")


def _unpack_batch(batch: Tuple[torch.Tensor, ...], device: str) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if len(batch) == 3:
        x_batch, lengths, y_batch = batch
        return x_batch.to(device), lengths.to(device), y_batch.to(device)

    x_batch, y_batch = batch
    return x_batch.to(device), None, y_batch.to(device)


def _forward_model(model: nn.Module, x_batch: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    if lengths is None:
        return model(x_batch)
    return model(x_batch, lengths)


def build_model(config: Dict, embedding_matrix: torch.Tensor | None = None) -> nn.Module:
    model_cfg = config["models"]
    model_name = model_cfg["model_name"].lower()

    if model_name == "ann":
        ann_cfg = model_cfg["ann"]
        return ANNClassifier(
            input_dim=ann_cfg["input_dim"],
            hidden_dims=ann_cfg["hidden_dims"],
            output_dim=ann_cfg["output_dim"],
            dropout=ann_cfg["dropout"],
        )

    if model_name == "lstm":
        if embedding_matrix is None:
            raise ValueError("embedding_matrix is required when model_name is lstm.")
        lstm_cfg = model_cfg["lstm"]
        return LSTMClassifier(
            embedding_matrix=embedding_matrix,
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            output_dim=lstm_cfg["output_dim"],
            dropout=lstm_cfg.get("dropout", 0.0),
            bidirectional=lstm_cfg.get("bidirectional", False),
            freeze_embeddings=lstm_cfg.get("freeze_embeddings", True),
        )

    raise ValueError(f"Unsupported model_name: {model_name}. Supported models: ann, lstm.")


def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    training_cfg = config["training"]
    default_lr = training_cfg["learning_rate"]
    embedding_lr = training_cfg.get("embedding_learning_rate")
    weight_decay = training_cfg["weight_decay"]

    embedding = getattr(model, "embedding", None)
    if embedding is None or embedding_lr is None or not embedding.weight.requires_grad:
        return torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=default_lr,
            weight_decay=weight_decay,
        )

    embedding_param_ids = {id(param) for param in embedding.parameters()}
    non_embedding_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in embedding_param_ids
    ]

    return torch.optim.Adam(
        [
            {
                "params": [param for param in embedding.parameters() if param.requires_grad],
                "lr": embedding_lr,
            },
            {
                "params": non_embedding_params,
                "lr": default_lr,
            },
        ],
        weight_decay=weight_decay,
    )


def train_baseline(config_path: str = "configs/experiment.yaml") -> Dict[str, list]:
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = resolve_device(config["training"]["device"])
    model_name = config["models"]["model_name"].lower()
    feature_type = _get_feature_type(config)
    artifact_name = _get_artifact_name(config)

    train_loader, val_loader, feature_encoder = prepare_dataloaders(config)
    embedding_matrix = None
    if model_name == "lstm":
        embedding_matrix = feature_encoder["embedding_matrix"]
    model = build_model(config, embedding_matrix=embedding_matrix).to(device)

    checkpoints_dir = Path("checkpoints")
    outputs_dir = Path("outputs")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    #0: neg, 1: neu, 2: pos cho weight balancing với inverse
    #11426 data, 5325 pos, 5643 neg, 458 neu
    #vậy weight pos = 0.715, neu = 8.315, 0.674
    class_weights = torch.tensor(config["training"]["class_weights"], dtype=torch.float32)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, config)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_epoch = -1

    tracking_cfg = config.get("tracking", {}).get("mlflow", {})
    logging_cfg = config.get("logging", {})
    env_mlflow_enabled = os.getenv("MLFLOW_ENABLED", "0") == "1"
    mlflow_enabled = bool(logging_cfg.get("enabled", False) or env_mlflow_enabled)
    should_log_mlflow = MLFLOW_AVAILABLE and mlflow_enabled
    if should_log_mlflow:
        mlflow.set_tracking_uri(tracking_cfg.get("tracking_uri", "file:./mlruns"))
        mlflow.set_experiment(tracking_cfg.get("experiment_name", "vsfc_ann_baseline"))

    epochs = config["training"]["epochs"]
    early_stopping_cfg = config["training"].get("early_stopping", {})
    early_stopping_enabled = early_stopping_cfg.get("enabled", False)
    early_stopping_patience = early_stopping_cfg.get("patience", 3)
    early_stopping_min_delta = early_stopping_cfg.get("min_delta", 0.001)
    early_stopping_fluctuation_delta = early_stopping_cfg.get("fluctuation_delta", 0.05)
    early_stopping_window = early_stopping_patience + 1
    epochs_without_meaningful_improvement = 0

    run_name = tracking_cfg.get("run_name", f"{model_name}_baseline")
    run_ctx = mlflow.start_run(run_name=run_name) if should_log_mlflow else None
    if should_log_mlflow:
        mlflow.log_params(flatten_dict(config))

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            x_batch, lengths, y_batch = _unpack_batch(batch, device)

            optimizer.zero_grad()
            logits = _forward_model(model, x_batch, lengths)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(y_batch.detach().cpu().numpy())

        train_loss = train_loss_sum / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)

        model.eval()
        val_loss_sum = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, lengths, y_batch = _unpack_batch(batch, device)

                logits = _forward_model(model, x_batch, lengths)
                loss = criterion(logits, y_batch)
                val_loss_sum += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.detach().cpu().numpy())
                val_targets.extend(y_batch.detach().cpu().numpy())

        val_loss = val_loss_sum / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_name": artifact_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "config": config,
        }
        epoch_ckpt_path = checkpoints_dir / f"{artifact_name}_epoch_{epoch + 1:02d}.pt"
        torch.save(checkpoint_payload, epoch_ckpt_path)

        has_meaningful_improvement = _is_meaningful_loss_improvement(
            val_loss,
            best_val_loss,
            early_stopping_min_delta,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(checkpoint_payload, checkpoints_dir / f"{artifact_name}_best_model.pt")

        if has_meaningful_improvement:
            epochs_without_meaningful_improvement = 0
        else:
            epochs_without_meaningful_improvement += 1

        is_plateauing = _is_loss_plateauing(
            history["val_loss"],
            early_stopping_window,
            early_stopping_fluctuation_delta,
        )

        if should_log_mlflow:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                },
                step=epoch + 1,
            )
        

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if early_stopping_enabled and not has_meaningful_improvement:
            print(
                "Early stopping monitor: "
                f"{epochs_without_meaningful_improvement}/{early_stopping_patience} "
                "epochs without validation-loss improvement. "
                f"Best Val Loss: {best_val_loss:.4f}"
            )

        if (
            early_stopping_enabled
            and (
                epochs_without_meaningful_improvement >= early_stopping_patience
                or is_plateauing
            )
        ):
            print(
                "Early stopping triggered: "
                f"validation loss did not improve by at least {early_stopping_min_delta:.4f} "
                f"for {early_stopping_patience} consecutive epochs, or plateaued within "
                f"{early_stopping_fluctuation_delta:.4f}. "
                f"Best epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}"
            )
            break

    epochs_ran = len(history["train_loss"])

    torch.save(
        {
            "epoch": epochs_ran,
            "model_name": artifact_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        checkpoints_dir / f"{artifact_name}_last_model.pt",
    )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{artifact_name}_baseline_model.pt"
    feature_encoder_paths = []

    torch.save(model.state_dict(), model_path)
    if model_name == "lstm":
        embedding_matrix_path = models_dir / f"{artifact_name}_embedding_matrix.pt"
        torch.save(feature_encoder["embedding_matrix"], embedding_matrix_path)
        feature_encoder_paths.append(embedding_matrix_path)

        word2vec_path = Path(feature_encoder["word2vec_path"])
        if word2vec_path.exists():
            feature_encoder_paths.append(word2vec_path)
    elif feature_type == "tfidf":
        feature_encoder_path = models_dir / f"{artifact_name}_tfidf_vectorizer.pkl"
        joblib.dump(feature_encoder, feature_encoder_path)
        feature_encoder_paths.append(feature_encoder_path)
    elif feature_type == "word2vec":
        word2vec_path = models_dir / f"{artifact_name}_word2vec_model.model"
        feature_encoder["word2vec"].save(str(word2vec_path))
        feature_encoder_paths.append(word2vec_path)

        if feature_encoder.get("tfidf") is not None:
            tfidf_path = models_dir / f"{artifact_name}_tfidf_weighting_vectorizer.pkl"
            joblib.dump(feature_encoder["tfidf"], tfidf_path)
            feature_encoder_paths.append(tfidf_path)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    print(f"Saved model to: {model_path}")
    for feature_encoder_path in feature_encoder_paths:
        print(f"Saved feature encoder to: {feature_encoder_path}")

    with open(outputs_dir / f"{artifact_name}_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    with open(outputs_dir / f"{artifact_name}_history.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f, sort_keys=False, allow_unicode=True)

    model.eval()
    final_val_preds, final_val_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            x_batch, lengths, y_batch = _unpack_batch(batch, device)
            logits = _forward_model(model, x_batch, lengths)
            preds = torch.argmax(logits, dim=1)
            final_val_preds.extend(preds.detach().cpu().numpy())
            final_val_targets.extend(y_batch.detach().cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        final_val_targets, final_val_preds, average="macro", zero_division=0
    )
    eval_metrics = {
        "final_val_accuracy": float(accuracy_score(final_val_targets, final_val_preds)),
        "final_val_precision": float(precision),
        "final_val_recall": float(recall),
        "final_val_f1": float(f1),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "confusion_matrix": confusion_matrix(final_val_targets, final_val_preds).tolist(),
    }

    with open(outputs_dir / f"{artifact_name}_metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(eval_metrics, f, sort_keys=False, allow_unicode=True)

    if should_log_mlflow:
        mlflow.log_metrics(
            {
                key: value
                for key, value in eval_metrics.items()
                if isinstance(value, (int, float))
            }
        )
        mlflow.log_artifact(str(outputs_dir / f"{artifact_name}_config.yaml"), artifact_path="outputs")
        mlflow.log_artifact(str(outputs_dir / f"{artifact_name}_history.yaml"), artifact_path="outputs")
        mlflow.log_artifact(str(outputs_dir / f"{artifact_name}_metrics.yaml"), artifact_path="outputs")
        mlflow.log_artifacts(str(checkpoints_dir), artifact_path="checkpoints")
        for feature_encoder_path in feature_encoder_paths:
            mlflow.log_artifact(str(feature_encoder_path), artifact_path="models")
        mlflow.pytorch.log_model(model.cpu(), artifact_path="model")

        epochs_axis = np.arange(1, len(history["train_loss"]) + 1)
        fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
        ax_loss.plot(epochs_axis, history["train_loss"], marker="o", label="Train Loss")
        ax_loss.plot(epochs_axis, history["val_loss"], marker="o", label="Val Loss")
        ax_loss.set_title("Training vs Validation Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        fig_loss.tight_layout()
        mlflow.log_figure(fig_loss, f"figures/{artifact_name}_loss_curve.png")
        plt.close(fig_loss)

        fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
        ax_acc.plot(epochs_axis, history["train_acc"], marker="o", label="Train Accuracy")
        ax_acc.plot(epochs_axis, history["val_acc"], marker="o", label="Val Accuracy")
        ax_acc.set_title("Training vs Validation Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        fig_acc.tight_layout()
        mlflow.log_figure(fig_acc, f"figures/{artifact_name}_accuracy_curve.png")
        plt.close(fig_acc)

        cm = confusion_matrix(final_val_targets, final_val_preds)
        fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Validation Confusion Matrix")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        fig_cm.tight_layout()
        mlflow.log_figure(fig_cm, f"figures/{artifact_name}_confusion_matrix.png")
        plt.close(fig_cm)

        mlflow.end_run()
    elif mlflow_enabled and not MLFLOW_AVAILABLE:
        print("MLflow logging is enabled in config, but mlflow package is not installed. Skipping MLflow.")

    print(f"Saved config/history to: {outputs_dir}")

    return history


if __name__ == "__main__":
    train_baseline("configs/experiment.yaml")
