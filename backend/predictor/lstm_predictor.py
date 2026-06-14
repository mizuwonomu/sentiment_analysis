import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.append(ROOT_DIR)

import torch
import joblib
import json

from src.data_loader import FeedbackProcessor
from src.trainer import build_model
from src.utils import load_config

CONFIG_PATH = "configs/lstm_w2vec_sota.yaml"
MODEL_PATH = "models/lstm_w2vec_seed42_best_model.pt"
EMBEDDING_PATH = "models/lstm_w2vec_seed42_embedding_matrix.pt"
VOCAB_PATH = "datasets/processed/vocab.json"

device = torch.device("cpu")

processor = FeedbackProcessor(max_length=50, device=device)
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    processor.vocab = json.load(f)

config = load_config(CONFIG_PATH)
embedding_matrix = torch.load(EMBEDDING_PATH, map_location=device)
model = build_model(config, embedding_matrix=embedding_matrix).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("--- Đã load LSTM thành công ---")
def encode(text: str):
    processed = processor.process_text(text)

    ids = [
        processor.vocab.get(word, processor.vocab["<UNK>"])
        for word in processed.split()
    ]

    length = min(len(ids), processor.max_length)

    padded = processor._pad_sequence(ids)

    return padded, length

def predict(text: str):
    x, length = encode(text)

    x_tensor = torch.tensor([x], dtype=torch.long).to(device)
    length_tensor = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x_tensor, length_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    return {
        "negative": float(probs[0]),
        "neutral": float(probs[1]),
        "positive": float(probs[2]),
    }

if __name__ == "__main__":
    text = "học sinh bình luận rằng thầy giáo dạy rất chán"
    result = predict(text)

    print("Text:", text)
    print("Result:", result)
