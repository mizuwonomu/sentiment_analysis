# Báo cáo so sánh multi-seed: ANN + TF-IDF và LSTM + Word2Vec

## 1. Mục tiêu thí nghiệm

Mục tiêu của thí nghiệm này là đánh giá độ ổn định và hiệu quả của hai mô hình chính trong bài toán phân loại cảm xúc tiếng Việt gồm ba lớp: negative, neutral và positive.

Hai mô hình được so sánh gồm:

- **ANN + TF-IDF**: mô hình dựa trên đặc trưng tần suất/tầm quan trọng của từ.
- **LSTM + Word2Vec**: mô hình tuần tự sử dụng embedding Word2Vec và có khả năng ghi nhớ thứ tự xuất hiện giữa các từ.

Cả hai mô hình đều được đánh giá lại theo cùng một quy trình huấn luyện có **early stopping** và cùng bốn random seeds:

```text
[67, 42, 36, 2026]
```

Việc chạy nhiều seed giúp kiểm tra xem kết quả có ổn định hay chỉ phụ thuộc vào một lần khởi tạo ngẫu nhiên may mắn.

## 2. Bảng so sánh kết quả validation trung bình qua 4 seeds

| Metric | ANN + TF-IDF | LSTM + Word2Vec | Chênh lệch LSTM - ANN |
|---|---:|---:|---:|
| Accuracy | 0.8929 ± 0.0024 | 0.8932 ± 0.0025 | 0.0003 |
| Macro Precision | 0.7483 ± 0.0025 | 0.7586 ± 0.0064 | 0.0103 |
| Macro Recall | 0.8208 ± 0.0077 | 0.8216 ± 0.0070 | 0.0008 |
| Macro F1 | 0.7706 ± 0.0019 | 0.7807 ± 0.0063 | 0.0101 |

## 3. Macro F1 theo từng seed

| Seed | ANN + TF-IDF Macro F1 | LSTM + Word2Vec Macro F1 |
|---:|---:|---:|
| 67 | 0.7716 | 0.7749 |
| 42 | 0.7725 | 0.7891 |
| 36 | 0.7701 | 0.7817 |
| 2026 | 0.7681 | 0.7769 |

## 4. Nhận xét chính

Nhìn chung, **LSTM + Word2Vec đạt kết quả tốt hơn ANN + TF-IDF ở macro F1**, với mức trung bình:

```text
ANN + TF-IDF:       0.7706 ± 0.0019
LSTM + Word2Vec:    0.7807 ± 0.0063
```

Mức cải thiện macro F1 trung bình là khoảng **0.0101**. Điều này cho thấy mô hình LSTM không chỉ tốt ở một seed riêng lẻ, mà vẫn giữ được lợi thế khi đánh giá qua nhiều lần khởi tạo khác nhau.

Về accuracy, hai mô hình gần như tương đương:

```text
ANN + TF-IDF:       0.8929 ± 0.0024
LSTM + Word2Vec:    0.8932 ± 0.0025
```

Chênh lệch accuracy chỉ khoảng **0.0003**, nên điểm khác biệt chính giữa hai mô hình không nằm ở tổng số dự đoán đúng, mà nằm ở khả năng cân bằng hiệu quả giữa các lớp.

## 5. Phân tích lớp neutral

Vì lớp neutral có số lượng mẫu ít hơn và khó phân loại hơn, macro F1 chịu ảnh hưởng khá nhiều từ khả năng xử lý lớp này.

Dựa trên confusion matrix cộng gộp qua 4 seeds, kết quả lớp neutral là:

| Model | Neutral Precision | Neutral Recall | Neutral F1 |
|---|---:|---:|---:|
| ANN + TF-IDF | 0.3652 | 0.6541 | 0.4687 |
| LSTM + Word2Vec | 0.4138 | 0.6575 | 0.5079 |

Kết quả này cho thấy LSTM + Word2Vec cải thiện rõ hơn ở lớp neutral, đặc biệt là **neutral precision** và **neutral F1**. Trong khi ANN + TF-IDF vẫn có khả năng phân loại tốt các lớp phổ biến như negative và positive, mô hình dựa trên tần suất từ gặp khó hơn với những câu có sắc thái trung tính hoặc phụ thuộc vào ngữ cảnh.

## 6. Diễn giải nguyên nhân

ANN + TF-IDF biểu diễn câu chủ yếu dựa trên tần suất và độ quan trọng của từ. Cách biểu diễn này mạnh với các tín hiệu cảm xúc rõ ràng, ví dụ những từ mang sắc thái trực tiếp như khen, chê, tốt, kém, hài lòng hoặc không hài lòng. Tuy nhiên, TF-IDF không mô hình hóa thứ tự từ trong câu, nên những cấu trúc có phụ thuộc ngữ cảnh hoặc đảo chiều cảm xúc có thể bị biểu diễn chưa đầy đủ.

Ngược lại, LSTM + Word2Vec đọc câu dưới dạng chuỗi token theo thứ tự. Nhờ đó, mô hình có khả năng học một phần quan hệ giữa các từ trong câu, ví dụ sự xuất hiện của phủ định, mức độ, hoặc các cụm từ thể hiện thái độ theo ngữ cảnh. Đây là lý do hợp lý khiến LSTM cải thiện macro F1, đặc biệt ở lớp neutral, nơi tín hiệu cảm xúc thường không rõ ràng như hai lớp negative và positive.

Nói cách khác, khi mô hình có khả năng ghi nhớ thứ tự giữa các từ, nó có thể biểu diễn cảm xúc của câu tốt hơn so với mô hình chỉ dựa trên tần suất xuất hiện của từ.

## 7. Kết luận

Sau khi đánh giá lại cả hai mô hình với cùng early stopping và cùng 4 random seeds, **LSTM + Word2Vec cho kết quả macro F1 trung bình cao hơn ANN + TF-IDF**:

```text
ANN + TF-IDF:       Macro F1 = 0.7706 ± 0.0019
LSTM + Word2Vec:    Macro F1 = 0.7807 ± 0.0063
```

Điều này cho thấy LSTM + Word2Vec là mô hình tốt hơn trong nhóm thí nghiệm hiện tại nếu tiêu chí chính là **macro F1**, tức khả năng cân bằng hiệu quả giữa các lớp. Tuy nhiên, mức chênh lệch accuracy giữa hai mô hình rất nhỏ, nên ANN + TF-IDF vẫn là một baseline mạnh, đơn giản và ổn định.

Kết luận phù hợp nhất nhóm kết luận là:

> ANN + TF-IDF là baseline mạnh cho các tín hiệu cảm xúc rõ ràng dựa trên từ vựng, nhưng LSTM + Word2Vec cho kết quả tốt hơn về macro F1 nhờ khả năng mô hình hóa thứ tự từ và ngữ cảnh, đặc biệt cải thiện hiệu quả ở lớp neutral.

## 8. Ghi chú về bước tiếp theo

Các kết quả trên mới được đánh giá trên validation set. Sau khi chốt cấu hình và quy trình đánh giá, cần thực hiện đánh giá cuối cùng trên test set một lần để kiểm tra khả năng tổng quát hóa của mô hình đã chọn.
