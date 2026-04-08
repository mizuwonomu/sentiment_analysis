import torch
import re
import json
import os
from datasets import load_dataset
from collections import Counter

class FeedbackProcessor:
    def __init__(self, max_length = 50, device="cuda"):
        self.max_length = max_length
        self.device = device
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}

    def load_data(self):
        print("---Đang tải datasets UIT ---")
        dataset = load_dataset("chapter544ou/vietnamese_students_feedback")
        dataset.save_to_disk("datasets/vsfc")

        #Có 3 phần data: train, valid và test
        #Mỗi phần có 2 cột chính: 'sentence' và 'sentiment' (0: Neg, 1: Neu, 2: Pos)
        self.train_raw = dataset['train']
        self.test_raw = dataset['test']
        print(f"Đã tải xong! Train: {len(self.train_raw)} câu.")

    def clean_text(self, text):
        text = text.lower()

        #Chỉ giữ lại chữ cái, số và một số dấu câu quan trọng cho cảm xúc
        text = re.sub(r'[^\w\s!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def build_vocab(self, min_freq = 1):
        print("--Xây dựng bộ vocab ---")
        all_words = []
        for item in self.train_raw:
            cleaned = self.clean_text(item['sentence'])
            all_words.extend(cleaned.split())
        
        #Đếm tần suất và chỉ giữ lại từ xuất hiện nhiều hơn min_freq (tránh stop words nhiều)
        word_counts = Counter(all_words)
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[word] = new_id
                self.id_to_word[new_id] = word
        
        print(f"Bộ từ vựng hiện tại có {len(self.vocab)} từ.")
    
    def _pad_sequence(self, sequence):
        if len(sequence) < self.max_length:
            return sequence + [self.vocab["<PAD>"]] * (self.max_length - len(sequence))
        else:
            return sequence[:self.max_length]
    
    def prepare_tensors(self, data_split):
        input_ids = []
        labels = []

        for item in data_split:
            cleaned = self.clean_text(item['sentence'])

            #Encode: chuyển chữ thành ID, không có trong vocab -> dùng <UNK>
            ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in cleaned.split()]

            #Padding and transition
            padded_ids = self._pad_sequence(ids)

            input_ids.append(padded_ids)
            labels.append(item['sentiment'])

        return (torch.tensor(input_ids).to(self.device),
                torch.tensor(labels).to(self.device))

    def save_processed(self, path="datasets/processed", X_train=None, y_train=None, x_test=None, y_test=None):
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(f"{path}/vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

        #Lưu tensors đã xử lý (x_train, y_train, x_test, y_test)
        data = {
            "X_train": X_train, "y_train": y_train,
            "x_test": x_test, "y_test": y_test
        }
        torch.save(data, f"{path}/tensors.pt")
        print(f"---Đã đóng gói dữ liệu tại {path}---")

    def load_processed(self, path="datasets/processed"):
        with open(f"{path}/vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.dump(f)
            self.id_to_word = {int(v): k for k,v in self.vocab.items()}

        data = torch.load(f"{path}/tensors.pt", map_location=self.device)
        print("---Đã load dữ liệu từ ổ cứng!! ----")
        return data["X_train"], data["y_train"], data["x_test"], data["y_test"]
