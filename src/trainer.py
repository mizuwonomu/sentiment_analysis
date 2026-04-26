import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, random_split

from .data_loader import FeedbackProcessor
from .models import ANNClassifier
from .utils import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def prepare_ann_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, TfidfVectorizer]:
    data_cfg = config["data"]
    training_cfg = config["training"]
    device = resolve_device(training_cfg["device"])
    max_features = data_cfg.get("max_features", data_cfg.get("max_Features"))

    processor = FeedbackProcessor(max_length=data_cfg["max_length"], device=device)
    processor.load_data()

    train_texts = [processor.process_text(item["sentence"]) for item in processor.train_raw]
    train_labels = [item["sentiment"] for item in processor.train_raw]

    tfidf = TfidfVectorizer(max_features=max_features)
    x_tfidf = tfidf.fit_transform(train_texts).toarray()

    x_tensor = torch.tensor(x_tfidf, dtype=torch.float32)
    y_tensor = torch.tensor(train_labels, dtype=torch.long)

    full_train_dataset = TensorDataset(x_tensor, y_tensor)

    train_loader = DataLoader(full_train_dataset, batch_size=training_cfg["batch_size"], shuffle=True)

    val_texts = [processor.process_text(item["sentence"]) for item in processor.val_raw]
    val_labels = [item["sentiment"] for item in processor.val_raw]
    x_val_tfidf = tfidf.transform(val_texts).toarray()
    x_val_tensor = torch.tensor(x_val_tfidf, dtype=torch.float32)
    y_val_tensor = torch.tensor(val_labels, dtype=torch.long)
    val_ds = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"], shuffle=False)

    return train_loader, val_loader, tfidf


def build_model(config: Dict) -> nn.Module:
    model_cfg = config["models"]
    model_name = model_cfg["model_name"].lower()

    if model_name != "ann":
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            "Only ANN is currently supported by this trainer."
        )

    ann_cfg = model_cfg["ann"]
    return ANNClassifier(
        input_dim=ann_cfg["input_dim"],
        hidden_dims=ann_cfg["hidden_dims"],
        output_dim=ann_cfg["output_dim"],
        dropout=ann_cfg["dropout"],
    )


def train_baseline(config_path: str = "configs/baseline.yaml") -> Dict[str, list]:
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = resolve_device(config["training"]["device"])
    model = build_model(config).to(device)
    model_name = config["models"]["model_name"].lower()

    train_loader, val_loader, tfidf = prepare_ann_dataloaders(config)

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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_epoch = -1

    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_preds, train_targets = [], []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
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
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
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
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "config": config,
        }
        epoch_ckpt_path = checkpoints_dir / f"{model_name}_epoch_{epoch + 1:02d}.pt"
        torch.save(checkpoint_payload, epoch_ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(checkpoint_payload, checkpoints_dir / f"{model_name}_best_model.pt")

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    torch.save(
        {
            "epoch": epochs,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        checkpoints_dir / f"{model_name}_last_model.pt",
    )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{model_name}_baseline_model.pt"
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"

    torch.save(model.state_dict(), model_path)
    joblib.dump(tfidf, vectorizer_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")

    with open(outputs_dir / f"{model_name}_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    with open(outputs_dir / f"{model_name}_history.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f, sort_keys=False, allow_unicode=True)

    print(f"Saved config/history to: {outputs_dir}")

    return history


if __name__ == "__main__":
    train_baseline("configs/baseline.yaml")
