import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

MAX_SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 2e-5


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        prompt = f"This is a comment: {text}"
        inputs = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        return input_ids, attention_mask, torch.tensor(label)


def load_data(filepath):
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data['comment'])
            labels.append(data['res_mark'])
    return texts, labels


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, patience, model_save_path):
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break


def main(filepath, model_save_path):
    texts, labels = load_data(filepath)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CommentDataset(texts_train, labels_train, tokenizer, MAX_SEQ_LEN)
    val_dataset = CommentDataset(texts_val, labels_val, tokenizer, MAX_SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, PATIENCE, model_save_path)



