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

from src.data_loader import FeedbackProcessor
from src.trainer import build_model
from src.utils import load_config

VECTORIZER_PATH = "models/ann_tfidf_seed42_tfidf_vectorizer.pkl"
CONFIG_PATH = "configs/ann_tfidf_sota.yaml"
MODEL_PATH = "models/ann_tfidf_seed42_best_model.pt"

device = torch.device("cpu")

processor = FeedbackProcessor(max_length= 50, device= device)

vectorizer = joblib.load(VECTORIZER_PATH)

config = load_config(CONFIG_PATH)

model = build_model(config).to(device)

checkpoint = torch.load(MODEL_PATH,map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("--- Đã load ANN và TF-IDF thành công ---")

def predict(text: str):
    processed = processor.process_text(text)

    x = torch.tensor(
        vectorizer.transform([processed]).toarray(),
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )[0]
    
    return {
        "negative": float(probs[0]),
        "neutral": float(probs[1]),
        "positive": float(probs[2]),
    }

if __name__ == "__main__":
    text = "môn này hay"
    result = predict(text)
    print(text)
    print(result)