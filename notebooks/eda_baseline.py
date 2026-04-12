import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Import FeedbackProcessor từ pipeline của Linh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import FeedbackProcessor

# BƯỚC 1: Load và tiền xử lý data qua pipeline
processor = FeedbackProcessor(
    max_length=50,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
processor.load_data()

print("--- Đang xử lý text qua pipeline của Linh ---")
train_texts = [processor.process_text(item['sentence']) for item in processor.train_raw]
train_labels = [item['sentiment'] for item in processor.train_raw]

# BƯỚC 2: TF-IDF vectorize
print("--- Đang vectorize bằng TF-IDF ---")
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(train_texts).toarray()  # sparse -> dense numpy array

print(f"Kích thước ma trận TF-IDF: {X_tfidf.shape}")  # (num_samples, 1000)

# BƯỚC 3: Thống kê EDA nhanh
from collections import Counter
label_counts = Counter(train_labels)
label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
print("\nThống kê nhãn:")
for k, v in sorted(label_counts.items()):
    print(f"  {label_names[k]}: {v} câu")

# BƯỚC 4: Chuyển sang PyTorch tensor
device = "cuda" if torch.cuda.is_available() else "cpu"
X_tensor = torch.tensor(X_tfidf, dtype=torch.float32).to(device)
y_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)

# Train/val split 80/20
dataset = TensorDataset(X_tensor, y_tensor)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# BƯỚC 5: Định nghĩa ANN model (PyTorch)
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
            # Không cần Softmax vì CrossEntropyLoss tự lo
        )

    def forward(self, x):
        return self.net(x)

model = ANN().to(device)
print(f"\n{model}")

# BƯỚC 6: Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
print("\n--- Bắt đầu train ---")
for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total

    # --- Validation ---
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# BƯỚC 7: Lưu model và vectorizer
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/baseline_ann_model.pt")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
print("\n--- Đã lưu model vào models/ thành công! ---")

# BƯỚC 8: Quick test inference
test_sentence = "Giảng viên nhiệt tình, bài giảng dễ hiểu"
processed = processor.process_text(test_sentence)
test_vec = torch.tensor(
    tfidf.transform([processed]).toarray(),
    dtype=torch.float32
).to(device)

model.eval()
with torch.no_grad():
    logits = model(test_vec)
    probs = torch.softmax(logits, dim=1)
    label_idx = probs.argmax().item()

labels = {0: "Tiêu cực (Negative)", 1: "Trung tính (Neutral)", 2: "Tích cực (Positive)"}
print(f"\nCâu test: {test_sentence}")
print(f"Kết quả: {labels[label_idx]} (Độ tin cậy: {probs.max().item()*100:.2f}%)")