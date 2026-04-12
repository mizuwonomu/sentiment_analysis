import pandas as pd

# 1. Đọc dữ liệu (Thay đường dẫn đến file .arrow hoặc .parquet của bạn)
# Vì Linh dùng cấu trúc của HuggingFace, bạn có thể dùng thư viện datasets để load cho nhanh
from datasets import load_from_disk

dataset = load_from_disk('datasets/vsfc')
train_data = pd.DataFrame(dataset['train'])

# 2. Xem 5 dòng đầu tiên
print("5 dòng đầu tiên của dữ liệu:")
print(train_data.head())

# 3. Kiểm tra độ lệch nhãn (Nhiệm vụ trọng tâm Tuần 1)
print("\nThống kê số lượng từng nhãn cảm xúc:")
print(train_data['sentiment'].value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer

# Khởi tạo bộ biến đổi TF-IDF
tfidf = TfidfVectorizer(max_features=1000) # Lấy 1000 từ phổ biến nhất

# Biến cột sentence thành ma trận số
X_tfidf = tfidf.fit_transform(train_data['sentence'])

print("\nKích thước ma trận TF-IDF của bạn:")
print(X_tfidf.shape) 
# Kết quả sẽ có dạng (số câu, 1000 đặc trưng)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Khởi tạo mô hình mạng nơ-ron (ANN)
model = Sequential([
    # Lớp đầu vào nhận 1000 đặc trưng từ TF-IDF
    Dense(512, activation='relu', input_shape=(1000,)), 
    Dropout(0.2), # Giúp mô hình không bị "học vẹt"
    
    # Lớp ẩn
    Dense(256, activation='relu'),
    Dropout(0.2),
    
    # Lớp đầu ra cho 3 loại cảm xúc (0: tiêu cực, 1: trung tính, 2: tích cực)
    Dense(3, activation='softmax') 
])

# Cấu hình việc học
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()
# Cho máy bắt đầu học (Huấn luyện)
# epochs=10 nghĩa là máy sẽ đọc đi đọc lại dữ liệu 10 lần
history = model.fit(X_tfidf.toarray(), train_data['sentiment'], 
                    epochs=10, 
                    batch_size=32, 
                    validation_split=0.2) # Trích 20% dữ liệu để tự kiểm tra
model.save('baseline_ann_model.h5')