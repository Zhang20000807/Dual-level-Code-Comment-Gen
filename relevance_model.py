import os
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

# 配置
input_folder = './output'  # JSON Lines 文件夹路径
model_name = 'microsoft/codebert-base'
max_length = 512  # 最大长度
batch_size = 8  # 批处理大小
num_epochs = 3  # 训练轮数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 tokenizer 和 CodeBERT 模型
tokenizer = RobertaTokenizer.from_pretrained(model_name)
config = RobertaConfig.from_pretrained(model_name)
codebert_model = RobertaModel.from_pretrained(model_name, config=config).to(device)


# 定义二分类层
class CodePairClassifier(nn.Module):
    def __init__(self, codebert_model, hidden_size=768):
        super(CodePairClassifier, self).__init__()
        self.codebert = codebert_model
        self.classifier = nn.Linear(hidden_size, 2)  # 二分类任务

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


model = CodePairClassifier(codebert_model).to(device)


# 定义数据集
class CodeDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.code_pairs = self.process_file(file_path)

    def process_file(self, file_path):
        pairs = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            record = json.loads(line)
            code_snippets = record.get('code_snippets', [])
            for snippet in code_snippets:
                code_summary = snippet.get('code_summary', '')
                code_snippet = snippet.get('code_snippet', '')

                # Generate positive pairs
                for other_snippet in code_snippets:
                    if snippet != other_snippet:
                        other_code_summary = other_snippet.get('code_summary', '')
                        other_code_snippet = other_snippet.get('code_snippet', '')
                        pairs.append((code_summary, code_snippet, 1))  # Positive pair
                        pairs.append((other_code_summary, other_code_snippet, 1))  # Positive pair
                        # Generate negative pairs
                        pairs.append((code_summary, other_code_snippet, 0))  # Negative pair
                        pairs.append((other_code_summary, code_snippet, 0))  # Negative pair

        return pairs

    def __len__(self):
        return len(self.code_pairs)

    def __getitem__(self, idx):
        code_summary, code_snippet, label = self.code_pairs[idx]

        # Tokenization
        code_summary_tokens = tokenizer.tokenize(code_summary)
        code_snippet_tokens = tokenizer.tokenize(code_snippet)

        # 拼接并确保长度
        tokens = ['<CLS>'] + code_summary_tokens + ['<SEP>'] + code_snippet_tokens + ['<EOS>']

        # 如果长度不满足最大长度，继续拼接
        while len(tokens) < self.max_length:
            for next_snippet in self.code_pairs:
                if next_snippet != (code_summary, code_snippet, label):  # 避免重复使用当前 snippet
                    next_code_summary, next_code_snippet, _ = next_snippet

                    # 对新 snippet 进行 tokenization
                    next_code_summary_tokens = tokenizer.tokenize(next_code_summary)
                    next_code_snippet_tokens = tokenizer.tokenize(next_code_snippet)

                    # 拼接新 snippet 的 tokens
                    tokens += next_code_summary_tokens + ['<SEP>'] + next_code_snippet_tokens
                    tokens = tokens[:self.max_length]  # 确保 tokens 长度不超过 max_length

                    if len(tokens) >= self.max_length:
                        break

            if len(tokens) >= self.max_length:
                break

        # 确保拼接后的文本符合最大长度
        tokens = tokens[:self.max_length]
        text = tokenizer.convert_tokens_to_string(tokens)
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                             return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(label)


# 训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Loss: {loss.item()}")

    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss for this epoch: {avg_loss}")


# 主要运行逻辑
def main():
    for filename in os.listdir(input_folder):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(input_folder, filename)
            dataset = CodeDataset(file_path, tokenizer, max_length)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 定义优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=1e-5)
            criterion = nn.CrossEntropyLoss()

            # 训练模型
            for epoch in range(num_epochs):
                print(f"Epoch {epoch + 1}/{num_epochs}")
                train(model, dataloader, optimizer, criterion)

            # 推断
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    input_ids, attention_mask, _ = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    logits = model(input_ids, attention_mask)
                    print(f"Results for {filename}:")
                    print(logits)


if __name__ == "__main__":
    main()
