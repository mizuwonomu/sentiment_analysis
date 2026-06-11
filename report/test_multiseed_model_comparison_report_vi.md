# Báo cáo đánh giá cuối trên tập test: ANN + TF-IDF và LSTM + Word2Vec

## 1. Mục tiêu

Báo cáo này tổng hợp kết quả đánh giá cuối cùng trên tập test của hai mô hình chính:

1. **ANN + TF-IDF**
2. **LSTM + Word2Vec**

Hai mô hình được đánh giá bằng cùng một protocol multi-seed với bốn seed:

```text
67, 42, 36, 2026
```

Mục tiêu của bước này là kiểm tra xem kết luận từ tập validation có còn giữ được trên tập test hay không. Tập test là dữ liệu chưa được sử dụng trong quá trình huấn luyện, tuning hyperparameter hoặc lựa chọn mô hình, vì vậy kết quả trên tập test phản ánh khả năng tổng quát hóa thực tế hơn.

---

## 2. Kết quả tổng hợp trên tập test

Bảng dưới đây trình bày kết quả trung bình và độ lệch chuẩn qua 4 seed.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| ANN + TF-IDF | 0.8628 ± 0.0023 | 0.7137 ± 0.0023 | 0.7629 ± 0.0035 | 0.7282 ± 0.0024 |
| LSTM + Word2Vec | **0.8707 ± 0.0019** | **0.7273 ± 0.0012** | **0.7868 ± 0.0063** | **0.7444 ± 0.0018** |

Chênh lệch trung bình giữa LSTM + Word2Vec và ANN + TF-IDF:

| Metric | Difference |
|---|---:|
| Accuracy | +0.0079 |
| Macro Precision | +0.0136 |
| Macro Recall | +0.0239 |
| Macro F1 | +0.0162 |

Kết quả cho thấy **LSTM + Word2Vec đạt kết quả cao hơn ANN + TF-IDF ở cả bốn metric chính**. Đặc biệt, macro F1 của LSTM cao hơn khoảng **0.0162**, trong khi độ lệch chuẩn của LSTM chỉ khoảng **0.0018**. Điều này cho thấy sự cải thiện không chỉ xuất hiện ở một seed riêng lẻ, mà tương đối ổn định qua nhiều lần khởi tạo ngẫu nhiên khác nhau.

---

## 3. Phân tích macro F1

Macro F1 là metric quan trọng trong bài toán này vì dataset có sự mất cân bằng giữa các lớp, đặc biệt là lớp **neutral** có số lượng mẫu ít hơn nhiều so với negative và positive. So với accuracy, macro F1 phản ánh tốt hơn khả năng học đều giữa các lớp.

Kết quả test macro F1:

| Model | Test Macro F1 |
|---|---:|
| ANN + TF-IDF | 0.7282 ± 0.0024 |
| LSTM + Word2Vec | **0.7444 ± 0.0018** |

LSTM + Word2Vec có macro F1 cao hơn ANN + TF-IDF và đồng thời có độ lệch chuẩn thấp hơn. Điều này cho thấy mô hình LSTM không chỉ đạt kết quả tốt hơn, mà còn ổn định hơn trong các lần chạy với seed khác nhau.

---

## 4. Phân tích lớp neutral

Vì lớp neutral là lớp khó nhất và có số lượng mẫu ít hơn, việc phân tích riêng lớp này giúp hiểu rõ hơn vì sao LSTM + Word2Vec đạt macro F1 tốt hơn.

Từ confusion matrix cộng gộp qua 4 seed, kết quả xấp xỉ cho lớp neutral như sau:

| Model | Neutral Precision | Neutral Recall | Neutral F1 |
|---|---:|---:|---:|
| ANN + TF-IDF | 0.305 | 0.525 | 0.386 |
| LSTM + Word2Vec | **0.328** | **0.588** | **0.421** |

LSTM + Word2Vec cải thiện rõ rệt ở lớp neutral:

| Metric | Improvement |
|---|---:|
| Neutral Precision | +0.023 |
| Neutral Recall | +0.063 |
| Neutral F1 | +0.035 |

Điều này cho thấy lợi thế macro F1 của LSTM chủ yếu đến từ khả năng xử lý lớp neutral tốt hơn. Đây là kết quả phù hợp với nhận xét từ tập validation: LSTM + Word2Vec không chỉ học tốt các lớp phổ biến như negative và positive, mà còn cải thiện khả năng nhận diện các phản hồi có sắc thái trung tính.

---

## 5. Nhận xét về đặc trưng TF-IDF và biểu diễn tuần tự của LSTM

ANN + TF-IDF là một baseline mạnh vì TF-IDF biểu diễn tốt các tín hiệu từ vựng rõ ràng. Với những câu có từ khóa cảm xúc trực tiếp, ví dụ các từ mang sắc thái tích cực hoặc tiêu cực rõ ràng, đặc trưng tần suất từ có thể hoạt động rất hiệu quả.

Tuy nhiên, TF-IDF về bản chất vẫn là biểu diễn dạng bag-of-words. Mô hình không trực tiếp ghi nhớ thứ tự xuất hiện giữa các từ, cũng như không biểu diễn rõ quan hệ ngữ cảnh giữa các token trong câu. Vì vậy, với những câu cần hiểu sắc thái, phủ định, mức độ, hoặc quan hệ giữa các cụm từ, TF-IDF có thể bị hạn chế.

Ngược lại, LSTM + Word2Vec xử lý câu như một chuỗi token. Mỗi từ được biểu diễn bằng vector Word2Vec, sau đó LSTM đọc lần lượt các vector này theo thứ tự xuất hiện trong câu. Nhờ đó, mô hình có khả năng học các mẫu tuần tự và ngữ cảnh giữa các từ tốt hơn so với mô hình chỉ dựa trên tần suất từ.

Trong thí nghiệm này, kết quả test cho thấy việc mô hình hóa thứ tự từ và ngữ cảnh giúp LSTM + Word2Vec đạt hiệu quả tốt hơn, đặc biệt ở macro F1 và lớp neutral. Điều này củng cố giả thuyết rằng với bài toán phân loại cảm xúc, biểu diễn tuần tự có thể giúp mô hình diễn đạt sắc thái cảm xúc tốt hơn so với đặc trưng tần suất từ đơn thuần.

---

## 6. Kết luận

Trên tập test, **LSTM + Word2Vec đạt kết quả tốt hơn ANN + TF-IDF ở cả accuracy, macro precision, macro recall và macro F1**. Đặc biệt, macro F1 của LSTM cao hơn khoảng **0.0162** và độ lệch chuẩn thấp hơn so với ANN, cho thấy kết quả này tương đối ổn định qua nhiều seed.

Kết quả này chứng minh rằng lợi thế của LSTM + Word2Vec không chỉ xuất hiện trên tập validation, mà còn được duy trì trên tập test chưa từng được sử dụng trong quá trình huấn luyện và tuning. Nói cách khác, trong thiết lập thí nghiệm hiện tại, LSTM + Word2Vec có khả năng tổng quát hóa tốt hơn ANN + TF-IDF.

Kết luận chính:

> Trong bài toán phân loại cảm xúc tiếng Việt này, mô hình LSTM + Word2Vec cho kết quả tổng quát hóa tốt hơn ANN + TF-IDF trên tập test. Lợi thế chính của LSTM đến từ khả năng mô hình hóa thứ tự từ và ngữ cảnh, giúp cải thiện hiệu quả phân loại theo macro F1, đặc biệt ở lớp neutral.

Tuy nhiên, ANN + TF-IDF vẫn là một baseline mạnh. Mô hình này đạt accuracy khá gần với LSTM và hoạt động tốt với các tín hiệu cảm xúc rõ ràng dựa trên từ vựng. Vì vậy, kết quả nên được diễn giải theo hướng: **LSTM + Word2Vec vượt trội hơn trong thiết lập thí nghiệm hiện tại, đặc biệt về khả năng cân bằng giữa các lớp**, thay vì kết luận rằng TF-IDF luôn kém trong mọi trường hợp.

