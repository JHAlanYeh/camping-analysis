import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import DataCollatorWithPadding
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import random
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import seaborn as sns

PRETRAINED_MODEL_NAME = "bert-base-multilingual-cased"  # 指定繁簡中文 MultilingualBERT-BASE 預訓練模型
NUM_LABELS = 3
random_seed = 82
result_text = ""

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

class MyDataset(Dataset):
    def __init__(self, df, mode ="train"):
        # tokenizer分词后可以被自动汇聚
        self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['content']]
        # Dataset会自动返回Tensor
        self.labels =  [label for label in df['label']]
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode != "test":
            return self.texts[idx], self.labels[idx]
        else:
            return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

class MultilingualBertClassifier(nn.Module):
    def __init__(self):
        super(MultilingualBertClassifier, self).__init__()
        self.model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME)
        self.pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)        
        self.dropout = nn.Dropout(0.5)        
        self.classifier = nn.Linear(self.config.hidden_size, NUM_LABELS)    

    def forward(self, input_id, mask):
        output_1 = self.model(input_ids=input_id, attention_mask=mask)        
        hidden_state = output_1[0]        
        pooler = hidden_state[:, 0]        
        pooler = self.pre_classifier(pooler)        
        pooler = torch.nn.ReLU()(pooler)        
        pooler = self.dropout(pooler)        
        output = self.classifier(pooler)        
        return output   

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, save_name):
    torch.save(model.state_dict(), f'new_data/docs_0724/Final_GPT4o/Type1_Result/MultilingualBERT/{save_name}')

def train_model():
    start_time = datetime.now()
    print(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    # 定义模型
    model = MultilingualBertClassifier()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)
    criterion = criterion.to(device)

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # 训练
    best_dev_acc = 0

    loss_list = []
    accuracy_list = []
    loss_val_list = []
    accuracy_val_list = []
    for epoch_num in range(epoch):
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += batch_loss.item()

        # ----------- 验证模型 -----------
        model.eval()
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for inputs, labels in tqdm(dev_loader):
                input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
                masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
                labels = labels.to(device)
                output = model(input_ids, masks)

                batch_loss = criterion(output, labels)
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc
                total_loss_val += batch_loss.item()

            training_info = f'''Epochs: {epoch_num + 1}
            | Train Loss: {total_loss_train / len(train_dataset): .3f}
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f}
            | Val Loss: {total_loss_val / len(dev_dataset): .3f}
            | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}\n'''
            print(training_info)

            save_result(training_info, "a+")
            save_result("=====================================\n", "a+")

            loss_list.append(total_loss_train / len(train_dataset))
            accuracy_list.append(100 * total_acc_train / len(train_dataset))
            loss_val_list.append(total_loss_val / len(dev_dataset))
            accuracy_val_list.append(100 * total_acc_val / len(dev_dataset))

            # 保存最优的模型
            print(f"total_acc_val / len(dev_dataset) = {'%.2f' % (total_acc_val / len(dev_dataset) * 100)}, best_dev_acc = {'%.2f' %  (best_dev_acc * 100)}")
            save_result(f"total_acc_val / len(dev_dataset) = {'%.2f' %  (total_acc_val / len(dev_dataset) * 100)}, best_dev_acc = {'%.2f' %  (best_dev_acc * 100)}\n", "a+")
            if total_acc_val / len(dev_dataset) > best_dev_acc:
                best_dev_acc = total_acc_val / len(dev_dataset)
                save_model(model, 'best.pt')
                best_epoch = epoch

        model.train()

    # 保存最后的模型，以便继续训练
    save_model(model, 'last.pt')
    # todo 保存优化器

    draw_acc_image(accuracy_list, accuracy_val_list)
    draw_loss_image(loss_list, loss_val_list)

    end_time = datetime.now()
    print(end_time.strftime("%Y-%m-%d %H:%M:%S"))

    total_time = end_time - start_time
    print(f"Total time:{total_time}")
    save_result("=====================================\n", "a+")
    save_result(f"Total time:{total_time}\n", "a+")
    save_result(f"Best Epoch:{best_epoch}\n", "a+")


def evaluate(dataset):
    # dataset = pd.read_csv("../model/origin_type1/test_df.csv").to_numpy()
    # 加载模型
    model = MultilingualBertClassifier()
    model.load_state_dict(torch.load('new_data/docs_0724/Final_GPT4o/Type1_Result/MultilingualBERT/best.pt'))
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=batch_size)
    total_acc_test = 0
    y_pred = []   #保存預測label
    y_true = []   #保存實際label
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            test_label = test_label.to(device)
            output = model(input_id, mask)
            _, preds = torch.max(output, 1)       
            y_pred.extend(preds.view(-1).detach().cpu().numpy())       # 將preds預測結果detach出來，並轉成numpy格式       
            y_true.extend(test_label.view(-1).detach().cpu().numpy())   
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(dataset): .3f}')
    cf_matrix = confusion_matrix(y_true, y_pred)
    show_confusion_matrix(y_true, y_pred, 3, "MultilingualBERT", epoch+1)
    print(accuracy_score(y_true, y_pred))
    # print(classification_report(y_true, y_pred, target_names=['負向', '中立' '正向'])) 
    print(cf_matrix)  
    print("scikit-learn Accuracy:", accuracy_score(y_true, y_pred))
    print("scikit-learn Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("scikit-learn Recall Score:", recall_score(y_true, y_pred, average="weighted"))
    print("scikit-learn F1 Score:", f1_score(y_true, y_pred, average="weighted"))

    save_result("=====================================\n", "a+")
    save_result("scikit-learn Accuracy:" + '%.2f' % (accuracy_score(y_true, y_pred) * 100) + "\n", "a+")
    save_result("scikit-learn Precision:" + '%.2f' % (precision_score(y_true, y_pred, average="weighted") * 100) + "\n", "a+")
    save_result("scikit-learn Recall Score:" + '%.2f' % (recall_score(y_true, y_pred, average="weighted") * 100) + "\n", "a+")
    save_result("scikit-learn F1 Score:" + '%.2f' % (f1_score(y_true, y_pred, average="weighted") * 100) + "\n", "a+")
    

def draw_loss_image(loss_list, loss_val_list):
    plt.figure()
    plt.plot(loss_list, label = 'train loss')
    plt.plot(loss_val_list, label = 'val loss')
    plt.title('MultilingualBERT Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoches')
    plt.legend()
    plt.savefig("new_data/docs_0724/Final_GPT4o/Type1_Result/MultilingualBERT/MultilingualBERT_Loss.jpg")

def draw_acc_image(accuracy_list, accuracy_val_list):
    plt.figure()
    plt.plot(accuracy_list, label = 'train acc')
    plt.plot(accuracy_val_list, label = 'val acc')
    plt.title('MultilingualBERT Training and validation acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoches')
    plt.legend()
    plt.savefig("new_data/docs_0724/Final_GPT4o/Type1_Result/MultilingualBERT/MultilingualBERT_Acc.jpg")

def show_confusion_matrix(y_true, y_pred, class_num, fname, epoch):
    cm = skm.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    labels = np.arange(class_num)
    sns.heatmap(
        cm, xticklabels=labels, yticklabels=labels,
        annot=True, linewidths=0.1, fmt='d', cmap='YlGnBu')
    plt.title(f'{fname} Confusion Matrix', fontsize=15)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    plt.savefig(fname=f"new_data/docs_0724/Final_GPT4o/Type1_Result/MultilingualBERT/{fname}.jpg")


def save_result(text, write_type):
    file_path = "new_data/docs_0724/Final_GPT4o/Type1_Result/MultilingualBERT/result.txt"
    open(file_path, write_type).close()
    with open(file_path, write_type) as f:
        f.write(text)
        f.close()


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.is_available())
    setup_seed(random_seed)

    df_train = pd.read_csv("new_data/docs_0724/Final_GPT4o/Type1_Result/gpt35_train_df.csv")
    df_val = pd.read_csv("new_data/docs_0724/Final_GPT4o/Type1_Result/val_df.csv")
    df_test = pd.read_csv("new_data/docs_0724/Final_GPT4o/Type1_Result/test_df.csv")

    # 因为要进行分词，此段运行较久，约40s
    train_dataset = MyDataset(df_train, "train")
    dev_dataset = MyDataset(df_val, "train")
    test_dataset = MyDataset(df_test, "test")

    print(len(df_train), len(dev_dataset), len(test_dataset))

    print("MultilingualBERT")
    print("=====================================")

    # 训练超参数
    save_result("MultilingualBERT", "w")
    save_result("\n=====================================\n", "a+")
    best_epoch = 0
    epoch = 10
    batch_size = 8
    lr = 2e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_model()
    evaluate(test_dataset)