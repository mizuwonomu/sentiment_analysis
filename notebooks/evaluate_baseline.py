import os
import sys
import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Import FeedbackProcessor và ANN từ pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import FeedbackProcessor

# Copy lại class ANN (phải giống hệt lúc train)
class ANN(nn.Module):
    def __init__(self, input_dim=1000, hidden1=512, hidden2=256, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# BƯỚC 1: Load model và TF-IDF đã lưu
device = "cuda" if torch.cuda.is_available() else "cpu"

tfidf = joblib.load("models/tfidf_vectorizer.pkl")

model = ANN().to(device)
model.load_state_dict(torch.load("models/baseline_ann_model.pt", map_location=device))
model.eval()

print("--- Đã load model và TF-IDF thành công ---")

# BƯỚC 2: Load test data qua đúng pipeline
processor = FeedbackProcessor(max_length=50, device=device)
processor.load_data()

print("--- Đang xử lý test data qua pipeline của Linh ---")
test_texts = [processor.process_text(item['sentence']) for item in processor.test_raw]
test_labels = [item['sentiment'] for item in processor.test_raw]

# BƯỚC 3: Vectorize + predict
X_test = torch.tensor(
    tfidf.transform(test_texts).toarray(),
    dtype=torch.float32
).to(device)

with torch.no_grad():
    logits = model(X_test)
    y_pred = logits.argmax(dim=1).cpu().numpy()

y_true = np.array(test_labels)

# BƯỚC 4: In kết quả
label_names = ["Negative", "Neutral", "Positive"]

print("\n--- KẾT QUẢ ĐÁNH GIÁ BASELINE (ANN + TF-IDF) ---")
print(classification_report(y_true, y_pred, target_names=label_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))