import pandas as pd
import numpy as np
import joblib
from datasets import load_from_disk
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# BƯỚC 1: Tải dữ liệu TEST (Phần này quan trọng để máy hiểu 'test_data' là gì)
dataset = load_from_disk('datasets/vsfc')
test_data = pd.DataFrame(dataset['test']) 
# BƯỚC 2: Tải bộ TF-IDF và Model đã lưu từ folder models
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
model = load_model('models/baseline_ann_model.h5')

# BƯỚC 3: Biến đổi dữ liệu test thành số
X_test_tfidf = tfidf.transform(test_data['sentence'])

# BƯỚC 4: Dự đoán
y_pred = model.predict(X_test_tfidf.toarray())
y_pred_labels = np.argmax(y_pred, axis=1)

# BƯỚC 5: In kết quả cuối cùng (Cái này để nộp báo cáo)
print("\n--- KẾT QUẢ ĐÁNH GIÁ BASELINE (ANN + TF-IDF) ---")
print(classification_report(test_data['sentiment'], y_pred_labels))
print("\nConfusion Matrix:")
print(confusion_matrix(test_data['sentiment'], y_pred_labels))
