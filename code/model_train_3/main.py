import torch
import numpy as np
import pandas as pd
import jieba
import random
import os
from datetime import datetime
from torch import nn
from sklearn.utils import shuffle
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, DistilBertTokenizer
from transformers import BertModel, BertConfig
from transformers import AutoModel, AutoConfig
from transformers import DistilBertModel, DistilBertConfig
from sklearn.utils.class_weight import compute_class_weight

LLM = "Breeze" # or "Origin", "Taide", "Breeze", "GPT4o", "GPT35", "TaiwanLLM"
CAMP_TYPE = "1" # or "2"
NUM_LABELS = 3
CURRENT_MODEL = ""

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 42
BEST_EPOCH = 0
TOTAL_EPOCH = 10
BATCH_SIZE = 16
LR = 2e-5

BERT_PRETRAINED_MODEL_NAME = "ckiplab/bert-base-chinese" 
ALBERT_PRETRAINED_MODEL_NAME = "ckiplab/albert-base-chinese" 
ROBERTA_PRETRAINED_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MULTILINGUAL_BERT_PRETRAINED_MODEL_NAME = "google-bert/bert-base-multilingual-cased"
DISTILBERT_PRETRAINED_MODEL_NAME = "Geotrend/distilbert-base-zh-cased"
CURRENT_PRETRAINED_MODEL_NAME = BERT_PRETRAINED_MODEL_NAME

class CampDataset(Dataset):
    def __init__(self, df, PRETRAINED_MODEL_NAME):
        if PRETRAINED_MODEL_NAME == DISTILBERT_PRETRAINED_MODEL_NAME:
            tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
        else:
            tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
        self.texts = [tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=512,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt') for text in df['text']]
  
        self.labels =  [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
    
class ModelClassifier(nn.Module):
    def __init__(self, PRETRAINED_MODEL_NAME):
        super(ModelClassifier, self).__init__()

        if PRETRAINED_MODEL_NAME == ALBERT_PRETRAINED_MODEL_NAME:
            self.model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
            self.config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
            self.dropout = nn.Dropout(0.5)        
            self.classifier = nn.Linear(self.config.hidden_size, NUM_LABELS)    
        elif PRETRAINED_MODEL_NAME == DISTILBERT_PRETRAINED_MODEL_NAME:
            self.model = DistilBertModel.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
            self.config = DistilBertConfig.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
            self.pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)        
            self.dropout = nn.Dropout(0.5)        
            self.classifier = nn.Linear(self.config.hidden_size, NUM_LABELS)  
        else:
            self.model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
            self.config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
            self.pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)        
            self.dropout = nn.Dropout(0.5)        
            self.classifier = nn.Linear(self.config.hidden_size, NUM_LABELS)   

    def forward(self, input_id, mask, PRETRAINED_MODEL_NAME):
        if PRETRAINED_MODEL_NAME == ALBERT_PRETRAINED_MODEL_NAME:
            output_1 = self.model(input_ids=input_id)        
            hidden_state = output_1[0]        
            pooler = hidden_state[:, 0]        
            pooler = nn.ReLU()(pooler)        
            pooler = self.dropout(pooler)        
            output = self.classifier(pooler)        
            return output   
        else:
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


def get_dataset(PRETRAINED_MODEL_NAME):
    train_dataset = CampDataset(df_train, PRETRAINED_MODEL_NAME)
    val_dataset = CampDataset(df_val, PRETRAINED_MODEL_NAME)
    test_dataset = CampDataset(df_test, PRETRAINED_MODEL_NAME)

    return train_dataset, val_dataset, test_dataset


def split_training_data():
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


    df = pd.read_csv(f"new_data/docs_0819/Final_Origin/type{CAMP_TYPE}_comments_origin.csv", encoding="utf_8_sig")
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

    np.random.seed(RANDOM_SEED)
    df_train, df_val, df_test = np.split(target_df.sample(frac=1, random_state=RANDOM_SEED), [int(.8*len(target_df)), int(.9*len(target_df))])
    print(len(df_train),len(df_val), len(df_test))

    df_train = shuffle(df_train)
    df_val = shuffle(df_val)
    df_test = shuffle(df_test)
    
    pd.DataFrame(df_train, columns=["content", "rating", "status", "type", "label", "sequence_num", "publishedDate", "origin", "text"]).to_csv(f"new_data/docs_0819/Final_Origin/type{CAMP_TYPE}_train_df.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(df_val, columns=["content", "rating", "status", "type", "label", "sequence_num", "publishedDate", "origin", "text"]).to_csv(f"new_data/docs_0819/Final_Origin/type{CAMP_TYPE}_val_df.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(df_test, columns=["content", "rating", "status", "type", "label", "sequence_num", "publishedDate", "origin", "text"]).to_csv(f"new_data/docs_0819/Final_Origin/type{CAMP_TYPE}_test_df.csv", index=False, encoding="utf-8-sig")

    return df_train, df_val, df_test

def save_model(model, save_name):
    torch.save(model.state_dict(), f'new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/{CURRENT_MODEL}/{NUM_LABELS}/{BATCH_SIZE}/{save_name}')

def train_model(PRETRAINED_MODEL_NAME):
    start_time = datetime.now()
    print(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    # 定义模型
    model = ModelClassifier(PRETRAINED_MODEL_NAME)
    # 定义损失函数和优化器

    criterion = nn.CrossEntropyLoss()
    if LLM == 'Origin':
        criterion = nn.CrossEntropyLoss(class_weights)
    
    optimizer = Adam(model.parameters(), lr=LR)
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 训练
    best_val_acc = 0

    loss_list = []
    accuracy_list = []
    loss_val_list = []
    accuracy_val_list = []
    for epoch_num in range(TOTAL_EPOCH):
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(DEVICE) # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(DEVICE) # torch.Size([32, 1, 35])
            labels = labels.to(DEVICE)
            output = model(input_ids, masks, CURRENT_PRETRAINED_MODEL_NAME)

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
            for inputs, labels in tqdm(val_loader):
                input_ids = inputs['input_ids'].squeeze(1).to(DEVICE)
                masks = inputs['attention_mask'].to(DEVICE)
                labels = labels.to(DEVICE)
                output = model(input_ids, masks ,CURRENT_PRETRAINED_MODEL_NAME)

                batch_loss = criterion(output, labels)
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc
                total_loss_val += batch_loss.item()

            training_info = f'''Epochs: {epoch_num + 1}
            | Train Loss: {total_loss_train / len(train_dataset): .3f}
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f}
            | Val Loss: {total_loss_val / len(val_dataset): .3f}
            | Val Accuracy: {total_acc_val / len(val_dataset): .3f}\n'''
            print(training_info)

            save_result(training_info, "a+")
            save_result("=====================================\n", "a+")

            loss_list.append(total_loss_train / len(train_dataset))
            accuracy_list.append(100 * total_acc_train / len(train_dataset))
            loss_val_list.append(total_loss_val / len(val_dataset))
            accuracy_val_list.append(100 * total_acc_val / len(val_dataset))

            # 保存最优的模型
            if total_acc_val / len(val_dataset) > best_val_acc:
                best_val_acc = total_acc_val / len(val_dataset)
                save_model(model, 'best.pt')
                best_epoch = epoch_num + 1
            print(f"total_acc_val / len(val_dataset) = {'%.2f' % (total_acc_val / len(val_dataset) * 100)}, best_dev_acc = {'%.2f' %  (best_val_acc * 100)}")
            save_result(f"total_acc_val / len(val_dataset) = {'%.2f' %  (total_acc_val / len(val_dataset) * 100)}, best_dev_acc = {'%.2f' %  (best_val_acc * 100)}\n", "a+")
           
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


def evaluate(dataset, PRETRAINED_MODEL_NAME):
    # 加载模型
    model = ModelClassifier(PRETRAINED_MODEL_NAME)
    model.load_state_dict(torch.load(f'new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/{CURRENT_MODEL}/{NUM_LABELS}/{BATCH_SIZE}/best.pt'))
    model = model.to(DEVICE)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    total_acc_test = 0
    y_pred = []   #保存預測label
    y_true = []   #保存實際label
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(DEVICE)
            mask = test_input['attention_mask'].to(DEVICE)
            test_label = test_label.to(DEVICE)
            output = model(input_id, mask, PRETRAINED_MODEL_NAME)
            _, preds = torch.max(output, 1)       
            y_pred.extend(preds.view(-1).detach().cpu().numpy())       # 將preds預測結果detach出來，並轉成numpy格式       
            y_true.extend(test_label.view(-1).detach().cpu().numpy())   
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(dataset): .3f}')
    cf_matrix = confusion_matrix(y_true, y_pred)
    show_confusion_matrix(y_true, y_pred, NUM_LABELS)
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
    plt.title(f'{CURRENT_MODEL} Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoches')
    plt.legend()
    plt.savefig(f"new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/{CURRENT_MODEL}/{NUM_LABELS}/{BATCH_SIZE}/{CURRENT_MODEL}_Loss.jpg")

def draw_acc_image(accuracy_list, accuracy_val_list):
    plt.figure()
    plt.plot(accuracy_list, label = 'train acc')
    plt.plot(accuracy_val_list, label = 'val acc')
    plt.title(f'{CURRENT_MODEL} Training and validation acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoches')
    plt.legend()
    plt.savefig(f"new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/{CURRENT_MODEL}/{NUM_LABELS}/{BATCH_SIZE}/{CURRENT_MODEL}_Acc.jpg")

def show_confusion_matrix(y_true, y_pred, class_num):
    cm = skm.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    labels = np.arange(class_num)
    sns.heatmap(
        cm, xticklabels=labels, yticklabels=labels,
        annot=True, linewidths=0.1, fmt='d', cmap='YlGnBu')
    plt.title(f'{CURRENT_MODEL} Confusion Matrix', fontsize=15)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    plt.savefig(fname=f"new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/{CURRENT_MODEL}/{NUM_LABELS}/{BATCH_SIZE}/{CURRENT_MODEL}.jpg")


def save_result(text, write_type):
    file_path = f"new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/{CURRENT_MODEL}/{NUM_LABELS}/{BATCH_SIZE}/result.txt"
    open(file_path, write_type).close()
    with open(file_path, write_type) as f:
        f.write(text)
        f.close()

def create_folder():
    file_path = f"new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/{CURRENT_MODEL}/{NUM_LABELS}/{BATCH_SIZE}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)

if __name__ == "__main__":
    setup_seed(RANDOM_SEED)

    df_train = pd.read_csv(f"new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/{LLM.lower()}_type{CAMP_TYPE}_train_df.csv")
    df_val = pd.read_csv(f"new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/type{CAMP_TYPE}_val_df.csv")
    df_test = pd.read_csv(f"new_data/docs_0819/Final_{LLM}/Type{CAMP_TYPE}_Result/type{CAMP_TYPE}_test_df.csv")

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df_train['label']), y=df_train['label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    ALL_MODEL = [ROBERTA_PRETRAINED_MODEL_NAME, MULTILINGUAL_BERT_PRETRAINED_MODEL_NAME, DISTILBERT_PRETRAINED_MODEL_NAME, BERT_PRETRAINED_MODEL_NAME, ALBERT_PRETRAINED_MODEL_NAME]
   
    for model, pretrained_model_name in  zip(["RoBERTa", "MultilingualBERT", "DistilBERT", "BERT", "ALBERT"], ALL_MODEL):
        CURRENT_MODEL = model
        CURRENT_PRETRAINED_MODEL_NAME = pretrained_model_name
        
        for bs in [8, 16]:
            BATCH_SIZE = bs

            print(CURRENT_MODEL)
            print(CURRENT_PRETRAINED_MODEL_NAME)
            print("=====================================")

            train_dataset, val_dataset, test_dataset = get_dataset(CURRENT_PRETRAINED_MODEL_NAME)

            create_folder()

            save_result(CURRENT_MODEL, "w")
            save_result(f"\nPretrained Model={CURRENT_PRETRAINED_MODEL_NAME}\n", "a+")
            save_result("\n=====================================\n", "a+")
            save_result(f"epoch={TOTAL_EPOCH}\n", "a+")
            save_result(f"batch_size={BATCH_SIZE}\n", "a+")
            save_result(f"lr={LR}\n", "a+")
            save_result("\n=====================================\n", "a+")

            train_model(CURRENT_PRETRAINED_MODEL_NAME)
            evaluate(test_dataset, CURRENT_PRETRAINED_MODEL_NAME)