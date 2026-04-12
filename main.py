from src.data_loader import FeedbackProcessor
import os
import torch

processed_path = "datasets/processed"
processor = FeedbackProcessor(max_length=50, device="cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(f"{processed_path}/tensors.pt"):
    X_train, y_train, x_test, y_test = processor.load_processed(processed_path)

else:
    processor.load_data()
    processor.build_vocab(min_freq=2)
    X_train, y_train = processor.prepare_tensors(processor.train_raw)
    x_test, y_test = processor.prepare_tensors(processor.test_raw)

    processor.save_processed(processed_path, X_train=X_train, y_train=y_train, x_test=x_test, y_test=y_test)

print(f"Sẵn sàng để train! Shape: {X_train.shape}")