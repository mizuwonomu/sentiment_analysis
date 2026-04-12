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