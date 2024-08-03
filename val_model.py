import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lstm.lstm import LSTMClassifier, load_data, build_vocab
from transformers import BertForSequenceClassification, BertTokenizer
from cnn.cnn import CNNClassifier
from bert.bert import CommentDataset as BertDataset
from lstm.lstm import CommentDataset as LSTMDataset
from cnn.cnn import CommentDataset as CNNDataset

BERT_DIR = ''
tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
output_dim = 2
file_path = ''
lstm_model_save_path = ''
cnn_model_save_path = ''
bert_model_save_path = ''
LSTM_setting = {
    'MAX_SEQ_LEN': 100,
    'BATCH_SIZE': 32,
    'EMBEDDING_DIM': 128,
    'HIDDEN_DIM': 128,
    'EPOCHS': 100,
}
CNN_setting = {
    'MAX_SEQ_LEN': 100,
    'BATCH_SIZE': 32,
    'EMBEDDING_DIM': 128,
    'NUM_FILTERS': 100,
    'KERNEL_SIZE': 3,
    'EPOCHS': 50,
    'PATIENCE': 5,
}
BERT_setting = {
    'MAX_SEQ_LEN': 128,
    'BATCH_SIZE': 16,
    'EPOCHS': 5,
    'PATIENCE': 2,
    'LEARNING_RATE': 2e-5,
}
model_names = {
    "LSTM": [lstm_model_save_path, LSTM_setting, LSTMDataset],
    "CNN": [cnn_model_save_path, CNN_setting, CNNDataset],
    "BERT": [bert_model_save_path, BERT_setting, BertDataset],
}
all_names = ['LSTM', 'CNN', 'BERT']


def load_dataset(comment_dataset, vocab, max_seq_len, batch_size, test_size=0.2):
    texts, labels = load_data(file_path)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=test_size, random_state=42)
    if model_name != "BERT":
        train_dataset = comment_dataset(texts_train, labels_train, vocab, max_seq_len)
        val_dataset = comment_dataset(texts_val, labels_val, vocab, max_seq_len)
    else:
        train_dataset = comment_dataset(texts_train, labels_train, tokenizer, max_seq_len)
        val_dataset = comment_dataset(texts_val, labels_val, tokenizer, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, label_encoder


def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    if model_name != "BERT":
        with torch.no_grad():
            for input_ids, labels in data_loader:
                outputs = model(input_ids)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
    else:
        with torch.no_grad():
            for input_ids, attention_mask, labels in data_loader:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                if total % 1000 == 0:
                    print(f"total: {total}")
                correct += (predicted == labels).sum().item()
                true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score


def evaluate_model(model_save_path, train_loader, val_loader, vocab_size):
    if model_name == "LSTM":
        model = LSTMClassifier(vocab_size, LSTM_setting['EMBEDDING_DIM'], LSTM_setting['HIDDEN_DIM'], output_dim)
    elif model_name == "CNN":
        model = CNNClassifier(vocab_size, CNN_setting['EMBEDDING_DIM'], CNN_setting['NUM_FILTERS'], CNN_setting['KERNEL_SIZE'], output_dim)
    elif model_name == "BERT":
        model = BertForSequenceClassification.from_pretrained(BERT_DIR, num_labels=2)
    else:
        exit(1)
    model.load_state_dict(torch.load(model_save_path))
    train_accuracy, train_precision, train_recall, train_f1_score = compute_accuracy(model, train_loader)
    val_accuracy, val_precision, val_recall, val_f1_score = compute_accuracy(model, val_loader)
    print(f'{model_name} Training Accuracy   : {train_accuracy:.4f}')
    print(f'{model_name} Training Precision  : {train_precision:.4f}')
    print(f'{model_name} Training Recall     : {train_recall:.4f}')
    print(f'{model_name} Training F1         : {train_f1_score:.4f}')
    print(f'{model_name} Validation Accuracy : {val_accuracy:.4f}')
    print(f'{model_name} Validation Precision: {val_precision:.4f}')
    print(f'{model_name} Validation Recall   : {val_recall:.4f}')
    print(f'{model_name} Validation F1       : {val_f1_score:.4f}')
    print()


def main():
    model_save_path, model_setting, comment_dataset = model_names[model_name]
    texts, _ = load_data(file_path)
    vocab = build_vocab(texts)
    train_loader, val_loader, _ = load_dataset(comment_dataset, vocab, model_setting['MAX_SEQ_LEN'], model_setting['BATCH_SIZE'])
    evaluate_model(model_save_path, train_loader, val_loader, len(vocab))


if __name__ == '__main__':
    for i, model_name in enumerate(all_names):
        main()
