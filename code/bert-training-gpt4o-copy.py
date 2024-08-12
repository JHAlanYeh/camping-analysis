from transformers import BertTokenizer, BertForSequenceClassification

# 加載預訓練的 BERT 分詞器和模型
PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=3)  # 假設是二分類任務

from torch.utils.data import DataLoader, Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 範例數據
import pandas as pd
from torch.utils.data import WeightedRandomSampler

df_train = pd.read_csv("new_data/docs_0804/Final_GPT4o/gpt4o_type1_merge_train_df_3_20240811.csv")
df_val = pd.read_csv("new_data/docs_0804/Final_Origin/Type1_Result/val_df_3.csv")
df_test = pd.read_csv("new_data/docs_0804/Final_Origin/Type1_Result/test_df_3.csv")
df_test = pd.concat([df_test, df_val])

weights = [0.2 if source == 0 else 0.8 for source in df_train['origin']]


train_dataset = CustomDataset(
    texts=[text for text in df_train['text']],
    labels=[text for text in df_train['label']],
    tokenizer=tokenizer,
    max_len=512
)

test_dataset = CustomDataset(
    texts=[text for text in df_test['text']],
    labels=[text for text in df_test['label']],
    tokenizer=tokenizer,
    max_len=512
)

sampler = WeightedRandomSampler(weights, num_samples=len(weights))
train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

from torch.optim import AdamW
from tqdm import tqdm

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定義優化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 訓練函數
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 開始訓練
epochs = 1
for epoch in range(epochs):
    loss = train_epoch(model, train_dataloader, optimizer, device)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')


from sklearn.metrics import accuracy_score

def eval_model(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device).unsqueeze(0)
            attention_mask = batch['attention_mask'].to(device).unsqueeze(0)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return accuracy_score(true_labels, predictions)

accuracy = eval_model(model, test_dataset, device)
print(f'Validation Accuracy: {accuracy:.4f}')
