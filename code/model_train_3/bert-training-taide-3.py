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

import jieba
jieba.load_userdict('code\\custom_dict.txt')
jieba.set_dictionary('code\\dict.txt.big')

f = open('code\\stopwords_zh_TW.dat.txt', encoding="utf-8")
STOP_WORDS = []
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))

f = open('code\\stopwords.txt', encoding="utf-8")
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))

# https://blog.csdn.net/qq_43426908/article/details/135342646

PRETRAINED_MODEL_NAME = "ckiplab/bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型
NUM_LABELS = 3
random_seed = 42
result_text = ""

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

class MyDataset(Dataset):
    def __init__(self, df, mode ="train"):
        # tokenizer分词后可以被自动汇聚
        if mode == "train":
            self.texts = [tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            # max_length=512,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt') for text in df['text']]
        else:
            self.texts = [tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        # max_length=512,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt') for text in df['text']]
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
        self.pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)        
        self.dropout = nn.Dropout(0.5)        
        self.classifier = nn.Linear(self.config.hidden_size, NUM_LABELS)   

    def forward(self, input_id, mask):
        output_1 = self.model(input_ids=input_id, attention_mask=mask)        
        hidden_state = output_1[0]        
        pooler = hidden_state[:, 0]        
        pooler = self.pre_classifier(pooler)        
        pooler = nn.ReLU()(pooler) 
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
    torch.save(model.state_dict(), f'new_data/docs_0819/Final_Taide/Type1_Result/BERT/{NUM_LABELS}/{save_name}')

def train_model():
    start_time = datetime.now()
    print(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    # 定义模型
    model = BertClassifier()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, eps=eps)
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
            if total_acc_val / len(dev_dataset) > best_dev_acc:
                best_dev_acc = total_acc_val / len(dev_dataset)
                save_model(model, 'best.pt')
                best_epoch = epoch
            print(f"total_acc_val / len(dev_dataset) = {'%.2f' % (total_acc_val / len(dev_dataset) * 100)}, best_dev_acc = {'%.2f' %  (best_dev_acc * 100)}")
            save_result(f"total_acc_val / len(dev_dataset) = {'%.2f' %  (total_acc_val / len(dev_dataset) * 100)}, best_dev_acc = {'%.2f' %  (best_dev_acc * 100)}\n", "a+")
           
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
    model = BertClassifier()
    model.load_state_dict(torch.load(f'new_data/docs_0819/Final_Taide/Type1_Result/BERT/{NUM_LABELS}/best.pt'))
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
    show_confusion_matrix(y_true, y_pred, NUM_LABELS, "BERT", epoch+1)
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
    

def preprocess_data():
    df = pd.read_csv("new_data/docs_0819/Final_Taide/type1_comments_origin.csv", encoding="utf_8_sig")
    target_df = df

    texts = []
    origins = []
    for row, origin in zip(target_df['content'], target_df['origin']):
        ws = jieba.cut(row, cut_all=False)
        new_ws = []
        for word in ws:
            if word not in STOP_WORDS:
                new_ws.append(word)
        text = "".join(new_ws)
        texts.append(text)

        if origin == 0:
            origins.append(0)
        else:
            origins.append(1)


    target_df["text"] = texts
    target_df["origin"] = origins

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

    np.random.seed(random_seed)
    df_train, df_val, df_test = np.split(target_df.sample(frac=1, random_state=random_seed), [int(.8*len(target_df)), int(.9*len(target_df))])
    print(len(df_train),len(df_val), len(df_test))

    df_train = shuffle(df_train)
    df_val = shuffle(df_val)
    df_test = shuffle(df_test)
    
    pd.DataFrame(df_train, columns=["content", "rating", "status", "type", "label", "sequence_num", "publishedDate", "origin", "text"]).to_csv("new_data/docs_0819/Final_Taide/type1_train_df.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(df_val, columns=["content", "rating", "status", "type", "label", "sequence_num", "publishedDate", "origin", "text"]).to_csv("new_data/docs_0819/Final_Taide/type1_val_df.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(df_test, columns=["content", "rating", "status", "type", "label", "sequence_num", "publishedDate", "origin", "text"]).to_csv("new_data/docs_0819/Final_Taide/type1_test_df.csv", index=False, encoding="utf-8-sig")

    return df_train, df_val, df_test

def draw_loss_image(loss_list, loss_val_list):
    plt.figure()
    plt.plot(loss_list, label = 'train loss')
    plt.plot(loss_val_list, label = 'val loss')
    plt.title('BERT Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoches')
    plt.legend()
    plt.savefig(f"new_data/docs_0819/Final_Taide/Type1_Result/BERT/{NUM_LABELS}/BERT_Loss.jpg")

def draw_acc_image(accuracy_list, accuracy_val_list):
    plt.figure()
    plt.plot(accuracy_list, label = 'train acc')
    plt.plot(accuracy_val_list, label = 'val acc')
    plt.title('BERT Training and validation acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoches')
    plt.legend()
    plt.savefig(f"new_data/docs_0819/Final_Taide/Type1_Result/BERT/{NUM_LABELS}/BERT_Acc.jpg")

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
    plt.savefig(fname=f"new_data/docs_0819/Final_Taide/Type1_Result/BERT/{NUM_LABELS}/{fname}.jpg")


def save_result(text, write_type):
    file_path = f"new_data/docs_0819/Final_Taide/Type1_Result/BERT/{NUM_LABELS}/result.txt"
    open(file_path, write_type).close()
    with open(file_path, write_type) as f:
        f.write(text)
        f.close()


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.is_available())
    setup_seed(random_seed)
    # df_train, df_val, df_test = preprocess_data()

    df_train = pd.read_csv("new_data/docs_0819/Final_Taide/Type1_Result/taide_type1_train_df.csv")
    df_val = pd.read_csv("new_data/docs_0819/Final_Taide/Type1_Result/type1_val_df.csv")
    df_test = pd.read_csv("new_data/docs_0819/Final_Taide/Type1_Result/type1_test_df.csv")

    # 因为要进行分词，此段运行较久，约40s
    train_dataset = MyDataset(df_train, "train")
    dev_dataset = MyDataset(df_val, "train")
    test_dataset = MyDataset(df_test, "test")

    print(len(df_train), len(dev_dataset), len(test_dataset))


    print("BERT")
    print("=====================================")

    # 训练超参数
    save_result("BERT", "w")
    save_result("\n=====================================\n", "a+")
    best_epoch = 0
    epoch = 10
    batch_size = 16
    lr = 2e-5
    eps = 1e-8

    save_result(f"epoch={epoch}\n", "a+")
    save_result(f"batch_size={batch_size}\n", "a+")
    save_result(f"lr={lr}\n", "a+")
    save_result("\n=====================================\n", "a+")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_model()
    evaluate(test_dataset)