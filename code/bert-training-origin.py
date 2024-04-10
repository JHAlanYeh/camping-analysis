import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import DataCollatorWithPadding
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import random
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

# https://blog.csdn.net/qq_43426908/article/details/135342646

PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型
NUM_LABELS = 3
random_seed = 1999

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

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.config.hidden_size, NUM_LABELS)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.model(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(random_seed)

def save_model(model, save_name):
    torch.save(model.state_dict(), f'../model/origin_type1/{save_name}')

def train_model():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # 定义模型
    model = BertClassifier()
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

            print(f'''Epochs: {epoch_num + 1}
            | Train Loss: {total_loss_train / len(train_dataset): .3f}
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f}
            | Val Loss: {total_loss_val / len(dev_dataset): .3f}
            | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}''')

            loss_list.append(total_loss_train / len(train_dataset))
            accuracy_list.append(100 * total_acc_train / len(train_dataset))
            loss_val_list.append(total_loss_val / len(dev_dataset))
            accuracy_val_list.append(100 * total_acc_val / len(dev_dataset))

            # 保存最优的模型
            if total_acc_val / len(dev_dataset) > best_dev_acc:
                best_dev_acc = total_acc_val / len(dev_dataset)
                save_model(model, 'best.pt')

        model.train()

    # 保存最后的模型，以便继续训练
    save_model(model, 'last.pt')
    # todo 保存优化器

    draw_acc_image(accuracy_list, accuracy_val_list)
    draw_loss_image(loss_list, loss_val_list)

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def evaluate(dataset):
    # dataset = pd.read_csv("../model/origin_type1/test_df.csv").to_numpy()
    # 加载模型
    model = BertClassifier()
    model.load_state_dict(torch.load('../model/origin_type1/best.pt'))
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
    print(cf_matrix)  
    print("scikit-learn precision:", precision_score(y_true, y_pred, average="weighted"))
    print("scikit-learn f1 score:", f1_score(y_true, y_pred, average="weighted"))
    print("scikit-learn recall score:", recall_score(y_true, y_pred, average="weighted"))
    print("scikit-learn Accuracy:", accuracy_score(y_true, y_pred))

def prprocess_data():
    min_num = 999999
    df1 = pd.read_csv("../docs/origin/type_origin.csv", encoding="utf_8_sig")

    df = df1[df1["content"].str.len() < 510]

    # for i in range(5):
    #     if i + 1 != 3:
    #         if min_num > len(df[df["rating"] == i + 1]):
    #             min_num = len(df[df["rating"] == i + 1])
    #     else:
    #         if min_num > len(df[df["rating"] == i + 1]):
    #             min_num = math.floor(len(df[df["rating"] == i + 1]) / 2)

    # clean_df = pd.DataFrame()
    # for i in range(5):
    #     if i + 1 <= 2:
    #         if len(df[(df["rating"] == i + 1) & (df["origin"] == 1)]) < min_num:
    #             clean_df = pd.concat([clean_df, df[(df["rating"] == i + 1) & (df["origin"] == 1)]])
    #             clean_df = pd.concat([clean_df, df[(df["rating"] == i + 1) & (df["origin"] == 0)].sample(n=min_num-len(df[(df["rating"] == i + 1) & (df["origin"] == 1)]))])
    #         else:
    #             clean_df = pd.concat([clean_df, df[(df["rating"] == i + 1) & (df["origin"] == 1)].sample(n=min_num)])
    #     elif i + 1 == 3:
    #         clean_df = pd.concat([clean_df, df[df["rating"] == i + 1].sample(n=min_num * 2)])
    #     else:
    #         clean_df = pd.concat([clean_df, df[df["rating"] == i + 1].sample(n=min_num)])
        

    target_df = df[["content", "status", "type"]]


    # create a list of our conditions
    conditions = [
        target_df['status'] == -1,
        target_df['status'] == 0,
        target_df['status'] == 1,
    ]

    # create a list of the values we want to assign for each condition
    values = [0, 1, 2]

    # create a new column and use np.select to assign values to it using our lists as arguments
    target_df['label'] = np.select(conditions, values)
    target_df = shuffle(target_df)
    # print(labels)

    np.random.seed(112)
    df_train, df_val, df_test = np.split(target_df.sample(frac=1, random_state=42), [int(.8*len(target_df)), int(.9*len(target_df))])
    print(len(df_train),len(df_val), len(df_test))

    return df_train, df_val, df_test

def draw_loss_image(loss_list, loss_val_list):
    plt.figure()
    plt.plot(loss_list, label = 'train loss')
    plt.plot(loss_val_list, label = 'val loss')
    plt.title('BERT Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoches')
    plt.legend()
    plt.savefig("../model/origin_type1/bert_loss.jpg")

def draw_acc_image(accuracy_list, accuracy_val_list):
    plt.figure()
    plt.plot(accuracy_list, label = 'train acc')
    plt.plot(accuracy_val_list, label = 'val acc')
    plt.title('BERT Training and validation acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoches')
    plt.legend()
    plt.savefig("../model/origin_type1/bert_acc.jpg")

if __name__ == "__main__":
    print(torch.__version__, torch.cuda.is_available())
    df_train, df_val, df_test = prprocess_data()
    

    # df_train = pd.read_csv("../model/origin_type1/train_df.csv")
    # df_val = pd.read_csv("../model/origin_type1/val_df.csv")
    # df_test = pd.read_csv("../model/origin_type1/test_df.csv")

    # 因为要进行分词，此段运行较久，约40s
    train_dataset = MyDataset(df_train, "train")
    dev_dataset = MyDataset(df_val, "train")
    test_dataset = MyDataset(df_test, "test")

    # pd.DataFrame(df_train, columns=["content", "status", "type", "label"]).to_csv("../model/origin_type1/train_df.csv")
    # pd.DataFrame(df_val, columns=["content", "status", "type", "label"]).to_csv("../model/origin_type1/val_df.csv")
    # pd.DataFrame(df_test, columns=["content", "status", "type", "label"]).to_csv("../model/origin_type1/test_df.csv")

    # 训练超参数
    epoch = 10
    batch_size = 8
    lr = 1e-5
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train_model()
    # evaluate(test_dataset)