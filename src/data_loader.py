import torch
import re
import json
import os
from datasets import load_dataset
from collections import Counter
from pyvi import ViTokenizer


class FeedbackProcessor:
    def __init__(self, max_length = 50, device="cuda"):
        self.max_length = max_length
        self.device = device
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}

        #load json viết tắt, abbreviation đã được có data chuẩn hoá từ trước
        base_dir = os.path.dirname(__file__)  # src/
        file_path = os.path.join(base_dir, "..", "datasets", "processed", "abbre.json")
        file_path = os.path.abspath(file_path)  # normalize path

        with open(file_path, "r", encoding="utf-8") as f:
            self.ABBREVIATIONS = json.load(f)

        #pattern -> viết tắt  
        self.pattern = re.compile(
            r'\b(' + '|'.join(map(re.escape, self.ABBREVIATIONS.keys())) + r')\b'  
        ) #kí hiệu \b: boundary, chỉ khớp nếu từ đó đứng độc lập (trước và sau nó là dấu câu hoặc space)

        #pattern -> dạng colon ()
        self.colon_pattern = re.compile(r'\b\w*colon\w*\b')              

    def load_data(self):
        print("---Đang tải datasets UIT ---")
        dataset = load_dataset("chapter544ou/vietnamese_students_feedback")
        dataset.save_to_disk("datasets/vsfc")

        #Có 3 phần data: train, valid và test
        #Mỗi phần có 2 cột chính: 'sentence' và 'sentiment' (0: Neg, 1: Neu, 2: Pos)
        self.train_raw = dataset['train']
        self.val_raw = dataset['validation']
        self.test_raw = dataset['test']
        print(f"Đã tải xong! Train: {len(self.train_raw)} câu.")

    def clean_text(self, text):
        text = text.lower()  

        #Chỉ giữ lại chữ cái, số và một số dấu câu quan trọng cho cảm xúc
        text = re.sub(r'([!?.,])', r' \1 ', text)       #LINH: tách kí tự đặc biệt khỏi chữ vd hello!! -> hello ! !
        text = re.sub(r'[^\w\s!?]', '', text) #chỉ giữ lại \w: word, \s: space, dấu ! và ?
        text = re.sub(r'\s+', ' ', text).strip() #loại đi nhiều hơn 1 khoảng trắng
        return text
    
    #Chuyển viết tắt -> thường -- LINH
    def normalize_abbreviation(self, text):                    
        return self.pattern.sub(lambda x: self.ABBREVIATIONS[x.group()], text)
    
    #remove emoji -- LINH
    def remove_emoji(self, text):
        return self.colon_pattern.sub('', text)
    
    #pyvi token các từ Việt -- LINH
    def tokenize(self, text):
        return ViTokenizer.tokenize(text)
    
    #làm gọn process -- LINH
    def process_text(self, text):
        text = self.remove_emoji(text)
        text = self.clean_text(text)
        text = self.normalize_abbreviation(text)
        
        return self.tokenize(text)

    def build_vocab(self, min_freq = 1):
        print("--Xây dựng bộ vocab ---")
        all_words = []
        for item in self.train_raw:
            processed = self.process_text(item['sentence'])
            all_words.extend(processed.split())
        
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
    
    def prepare_tensors(self, data_split, return_lengths=False):
        input_ids = []
        labels = []
        lengths = []

        for item in data_split:
            processed = self.process_text(item['sentence'])

            #Encode: chuyển chữ thành ID, không có trong vocab -> dùng <UNK>
            ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in processed.split()]
            length = min(len(ids), self.max_length)

            #Padding and transition
            padded_ids = self._pad_sequence(ids)

            input_ids.append(padded_ids)
            labels.append(item['sentiment'])
            lengths.append(length)

        tensors = (
            torch.tensor(input_ids, dtype=torch.long).to(self.device),
            torch.tensor(labels, dtype=torch.long).to(self.device),
        )
        if return_lengths:
            return (*tensors, torch.tensor(lengths, dtype=torch.long).to(self.device))
        return tensors

    def save_processed(
        self,
        path="datasets/processed",
        X_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        X_val=None,
        y_val=None,
    ):
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(f"{path}/vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

        #Lưu tensors đã xử lý (x_train, y_train, x_test, y_test)
        data = {
            "X_train": X_train, "y_train": y_train,
            "x_test": x_test, "y_test": y_test
        }
        if X_val is not None and y_val is not None:
            data["X_val"] = X_val
            data["y_val"] = y_val
        torch.save(data, f"{path}/tensors.pt")
        print(f"---Đã đóng gói dữ liệu tại {path}---")

    def load_processed(self, path="datasets/processed", include_validation=False):
        with open(f"{path}/vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            self.id_to_word = {int(v): k for k,v in self.vocab.items()}

        data = torch.load(f"{path}/tensors.pt", map_location=self.device)
        print("---Đã load dữ liệu từ ổ cứng!! ----")
        if include_validation:
            return (
                data["X_train"],
                data["y_train"],
                data["x_test"],
                data["y_test"],
                data.get("X_val"),
                data.get("y_val"),
            )
        return data["X_train"], data["y_train"], data["x_test"], data["y_test"]
