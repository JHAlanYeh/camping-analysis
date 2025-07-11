import os
import re
import time
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import requests
from selenium import webdriver
from bs4 import BeautifulSoup as Soup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from dateutil.relativedelta import relativedelta
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import json
import jieba
from torch import nn
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from wordcloud import WordCloud
import jieba.posseg as pseg

user_id = "U30bc9aaf24ea900745f69d036821e5e3"
channel_access_token = "kQ/vmM5sqBwC9Dyc4ODf1aD/CeTTHZKbU32UCZsxHKgsIglO7oqC29E6mgJcU8jpfT57f2Ordq0fglHqXXw33D5142fPiE6mCCT1m5PJ38Jnfu2qBhXp2m6q01jAZBI4LSc+qmSZzUze5AVUN4obEwdB04t89/1O/w1cDnyilFU="

browserOptions = webdriver.ChromeOptions()
browserOptions.add_argument("--start-maximized")

user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15"
browserOptions.add_argument('--user-agent={}'.format(user_agent))
# browserOptions.add_argument("--headless")
browserOptions.add_argument('--mute-audio')
# INFO_0 / WARNING_1 / ERROR_2 / FATAL_3 / DEFAULT_0
browserOptions.add_argument("log-level=3")
# ****************************************************************************** #

PRETRAINED_MODEL_NAME = "bert-base-chinese"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


class ModelClassifier(nn.Module):
    def __init__(self, PRETRAINED_MODEL_NAME):
        super(ModelClassifier, self).__init__()
       
        self.model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
        self.config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
        self.pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)        
        self.dropout = nn.Dropout(0.5)        
        self.classifier = nn.Linear(self.config.hidden_size, 3)   

    def forward(self, input_id, mask, PRETRAINED_MODEL_NAME):
        output_1 = self.model(input_ids=input_id, attention_mask=mask)        
        hidden_state = output_1[0]        
        pooler = hidden_state[:, 0]        
        pooler = self.pre_classifier(pooler)        
        pooler = nn.ReLU()(pooler) 
        pooler = self.dropout(pooler)        
        output = self.classifier(pooler)        
        return output 


current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

print(f"========== 現在時間：{current_time} ==========")
print("========== 收集評論 開始 ==========")

browser = webdriver.Chrome(options=browserOptions)
wait = WebDriverWait(browser, 20)
default_url = "https://www.google.com/maps?authuser=0"
browser.get(default_url)
house_type = False


wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#searchboxinput")))
browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys("蟬說：山中靜靜")
browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys(Keys.ENTER)

time.sleep(5)
try:
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div:nth-child(3) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db")))
    address = browser.find_element(By.CSS_SELECTOR, "div:nth-child(3) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db").text

    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".DUwDvf.lfPIob")))
    name = browser.find_element(By.CSS_SELECTOR, ".DUwDvf.lfPIob").text.replace(":", "_").replace("\\", "_").replace("/", "_").replace("|", "_")
    print(name)

 
except Exception as e:
    print(e.args)


google_map = browser.current_url

absolute_position = (google_map.split("@")[1]).split(",")
latitude = absolute_position[0]
longitude = absolute_position[1]

file_name = "C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-management\\public\\data-sources\\comments.json"
comment_objs = []
with open(file_name, 'r', encoding="utf-8-sig") as file:
    comment_objs = json.load(file)

time.sleep(5)
wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(2)")))
title = browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(2)").text
if title != "評論":
    title = browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(3)").text
    if title != "評論":
        print("no comments")
    browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(3)").click()
    house_type = True
else:
    browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(2)").click()
    house_type = False

time.sleep(5)

wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.jANrlb > div.fontBodySmall")))
reviews_count = browser.find_element(By.CSS_SELECTOR, "div.jANrlb > div.fontBodySmall").text.replace(" 篇評論", "").replace(",", "")
# print("總評論數：" + reviews_count)

# if int(reviews_count) > 5000:
#     print("comments too much")

current_reviews_count = 0

pane = browser.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
browser.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", pane)

try:
    # 等待網頁元素的出現
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.m6QErb.Pf6ghf.KoSBEe.ecceSd.tLjsW > div.TrU0dc.kdfrQc > button")))
    # 找到排序方法的按鈕
    browser.find_element(By.CSS_SELECTOR, "div.m6QErb.Pf6ghf.KoSBEe.ecceSd.tLjsW > div.TrU0dc.kdfrQc > button").click()
except Exception as e:
    # 等待網頁元素的出現
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf > div:nth-child(8) > button:nth-child(2)")))
    # 找到排序方法的按鈕
    browser.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf > div:nth-child(8) > button:nth-child(2)").click()


time.sleep(5)
wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#action-menu > div:nth-child(2)")))
browser.find_element(By.CSS_SELECTOR, '#action-menu > div:nth-child(2)').click()

while int(reviews_count) > current_reviews_count and current_reviews_count < 300:
    pane = browser.find_element(By.CSS_SELECTOR, "div:nth-child(2) > div > div.e07Vkf.kA9KIf > div > div > div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
    browser.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", pane)

    time.sleep(5)

    # 獲取網頁原始碼
    soup = Soup(browser.page_source, "html.parser")

    # 獲取評論資料框架
    all_reviews = soup.select(".jftiEf.fontBodyMedium")
    # ar = all_reviews[0] # 第幾則評論
    current_reviews_count = len(all_reviews)
    print("目前爬到評論數：" + str(current_reviews_count))

time.sleep(5)

wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".jftiEf.fontBodyMedium")))
all_reviews = browser.find_elements(By.CSS_SELECTOR, ".jftiEf.fontBodyMedium")
new_comments_cnt = 0
new_comments_predict = {"positive": 0, "neutral": 0, "negative": 0}
for ar in all_reviews:
    try:
        if house_type == False:
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".rsqaWe")))
            publishedDate = ar.find_element(By.CSS_SELECTOR, ".rsqaWe").text.replace("Google", "").replace("(","").replace(")","").strip()
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".kvMYJc")))
            star = ar.find_element(By.CSS_SELECTOR, ".kvMYJc").get_attribute("aria-label").replace(" 顆星", "").strip()
        else:
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".xRkPPb")))
            publishedDate = ar.find_element(By.CSS_SELECTOR, ".xRkPPb").text.replace("Google", "").replace("(","").replace(")","").strip()
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".fzvQIb")))
            star = ar.find_element(By.CSS_SELECTOR, ".fzvQIb").text.replace("/5", "").strip()
            author = ar.find_element(By.CSS_SELECTOR, ".d4r55").text.strip()
            author_id = ar.find_element(By.CSS_SELECTOR, ".al6Kxe").get_attribute("data-href").split("/")[5].strip()
    
        today = datetime.now()

        if "月" in publishedDate:
            num = publishedDate.replace("個月前", "").strip()
            createdDate = today - relativedelta(months=int(num))
        elif "天" in publishedDate:
            num = publishedDate.replace("天前", "").strip()
            createdDate = today - timedelta(days=int(num))
        elif "年" in publishedDate:
            num = publishedDate.replace("年前", "").strip()
            createdDate = today - relativedelta(years=int(num))
        elif "週" in publishedDate:
            num = publishedDate.replace("週前", "").strip()
            createdDate = today - timedelta(weeks=int(num))
        else:
            createdDate = today

        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".MyEned")))
        content = ar.find_element(By.CSS_SELECTOR, ".MyEned").text
        if "全文" in content:
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".MyEned > span > button")))
            ar.find_element(By.CSS_SELECTOR, ".MyEned > span > button").click()
            content = ar.find_element(By.CSS_SELECTOR, ".MyEned > .wiI7pd").text

        content = content.replace("\n", "").replace("\r", "").replace("\t", "")


        duplicate =  list(filter(lambda x: x['author_id'] == author_id, comment_objs))
        if len(duplicate) == 0:
            new_comments_cnt += 1
            comment_objs.append({
                "author_id": author_id,
                "author": author,
                "content": content,
                "publishedDate": createdDate.strftime("%Y/%m/%d"),
                "rating": int(star),
                "reply_comments": ""
            })
        else:
            print("duplicate")
            break
    except Exception as e:
        print(e.args)
        # print({
        #         "author_id": author_id,
        #         "author": author,
        #         "content": "",
        #         "publishedDate": createdDate.strftime("%Y/%m/%d"),
        #         "rating": int(star),
        #     })
        continue

browser.quit()

print("========== 收集評論 結束 ==========")
print(f"========== 新增評論數：{new_comments_cnt}")


print("========== 載入情緒分析模型 開始 ==========")
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, force_download=True)
model = ModelClassifier(PRETRAINED_MODEL_NAME)
model.load_state_dict(torch.load(f'new_data/docs_0819/Final_TaiwanLLM/Type2_Result/BERT/3/8/best.pt'))
model = model.to(DEVICE)
model.eval()
print("========== 載入情緒分析模型 結束 ==========")

print("========== 模型預測情緒正向/中立/負向 開始 ==========")
sorted_comments = sorted(comment_objs, key=lambda d: d["publishedDate"], reverse=True)

for sc in sorted_comments:
    published_date = datetime.strptime(sc["publishedDate"], "%Y/%m/%d")
    delta = datetime.now() - published_date
    days_difference = delta.days

    if days_difference > 365:
        sc["publishedDateDesc"] = str(days_difference // 365) + "年前"
    elif days_difference > 30:
        sc["publishedDateDesc"] = str(days_difference // 30) + "月前"
    elif days_difference > 7:
        sc["publishedDateDesc"] = str(days_difference // 7) + "週前"
    elif days_difference > 0:
        sc["publishedDateDesc"] = str(days_difference) + "天前"
    else:
        sc["publishedDateDesc"] = "今天"


    ws = jieba.cut(sc["content"], cut_all=False)
    new_ws = []
    for word in ws:
        if word not in STOP_WORDS:
            new_ws.append(word)
    sc["text"] = "".join(new_ws)

    if "predict" in sc:
        continue

    if sc["rating"] >= 4:
        sc["label"] = 2
    elif sc["rating"] == 3:
        sc["label"] = 1
    else:
        sc["label"] = 0

    text = tokenizer.encode_plus(
                        sc["text"],
                        add_special_tokens=True,
                        max_length=510,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt') 
  
    label =  sc["label"]

    with torch.no_grad():
        input_id = text['input_ids'].squeeze(1).to(DEVICE)
        mask = text['attention_mask'].to(DEVICE)
        output = model(input_id, mask, PRETRAINED_MODEL_NAME)
        _, preds = torch.max(output, 1)       
        y_pred = preds.view(-1).detach().cpu().numpy()
        # print(y_pred, label)

    sc["predict"] = int(y_pred[0])
    if sc["predict"] == 0:
        new_comments_predict["negative"] += 1
    elif sc["predict"] == 1:
        new_comments_predict["neutral"] += 1
    else:
        new_comments_predict["positive"] += 1

print("========== 模型預測情緒正向/中立/負向 結束 ==========")
print(f"========== 新增正向評論：{new_comments_predict['positive']}")
print(f"========== 新增中立評論：{new_comments_predict['neutral']}")
print(f"========== 新增負向評論：{new_comments_predict['negative']}")

print("========== 載入大型語言模型 Taiwan LLM 8B 開始 ==========")
model_id = "llm_model\Llama-3-Taiwan-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=nf4_config
    # attn_implementation="flash_attention_2" # optional
)
print("========== 載入大型語言模型 Taiwan LLM 8B 結束 ==========")

print("========== 自動生成回覆評論 開始 ==========")
# reply comments
for sc in comment_objs:
    if "reply_comments" not in sc or sc["reply_comments"] == "":
        messages = [
            {
                "role": "role", 
                "content": """你是一個經營露營地的人員，露營地名稱叫做「蟬說：山中靜靜」，請你根據貴賓的評論給予回覆，如果貴賓給予的是正面評價，請你回應貴賓感激的話，如果貴賓回覆的是負面評價，請你根據貴賓提及的問題，先道歉讓貴賓感受到誠意，接著提出未來改善的方向給貴賓。若你無法回答，請你說「很抱歉，我無法回答您的問題。」"""
            },
            {
                "role": "assistant", 
                "content": "好的，請您上傳露營的評論內容。"
            },
            {
                "role": "user", 
                "content": sc["content"]
            },
        ]

        
        input_ids  = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=8196,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        embeddings = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(embeddings, skip_special_tokens=True)
        # print(sc["content"])
        # print(response)
        if "很抱歉，我無法回答" in response:
            if sc["rating"] > 4:
                sc["reply_comments"] = "非常感謝您的讚賞與支持，蟬說：山中靜靜一直以來致力於提供優質的露營體驗，很高興聽到您對我們的設備、景色和美食都有良好的評價。您的回訪與口碑推薦對我們來說是莫大的鼓勵，我們會繼續努力，為您帶來更精彩的露營體驗。"
            else:
                sc["reply_comments"] = ""
            continue
        else:
            sc["reply_comments"] = response

        if sc["predict"] == 0 or sc["rating"] < 3:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + channel_access_token
            }

            push_text = f"""有一則緊急評論待處理，內容如下：\n{sc["content"]}\n============\n\n已自動回覆如下：\n{sc["reply_comments"]}\n\n如需更改回覆內容，請至後台進行修改。"""
            res = requests.post("https://api.line.me/v2/bot/message/push", json={"to": user_id, "messages": [{"type": "text", "text": push_text}]}, headers=headers)

    else:
        continue

print("========== 自動生成回覆評論 結束 ==========")

with open(file_name, 'w', encoding="utf-8-sig") as f:
    json.dump(sorted_comments, f, indent=4, ensure_ascii=False, sort_keys=False)
    print("save {file_name}".format(file_name=file_name))

with open('C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-demo\\data-sources\\comments.json', 'w', encoding="utf-8-sig") as f:
    json.dump(sorted_comments, f, indent=4, ensure_ascii=False, sort_keys=False)
    # print("save {file_name}".format(file_name=file_name))

# U30bc9aaf24ea900745f69d036821e5e3
# mu+Gm6HdSJ+aVfDjpd1X0DBiUUOipdTkSHDBJlF8AMM4Fpa3ThkAccgRS1ezP1ghfT57f2Ordq0fglHqXXw33D5142fPiE6mCCT1m5PJ38LFNxn/D1LJYtnYce1LIYwiuiqI2rE+2Pul67FexSCNjgdB04t89/1O/w1cDnyilFU=


print("========== 產生關鍵字文字雲 開始 ==========")

def wordcloud_generator(words, file_name):
    #文字雲造型圖片
    # mask = np.array(Image.open('picture.png')) #文字雲形狀
    # 從 Google 下載的中文字型
    font = 'SourceHanSansTW-Regular.otf'
    #背景顏色預設黑色，改為白色、使用指定圖形、使用指定字體
    my_wordcloud = WordCloud(width=1000, height=400,background_color='white', font_path=font).generate(words)
    plt.imshow(my_wordcloud)
    plt.axis("off")
    # plt.show()
    #存檔
    my_wordcloud.to_file(file_name)


file_name = "C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-management\\public\\data-sources\\comments.json"
comment_objs = []
with open(file_name, 'r', encoding="utf-8-sig") as file:
    comment_objs = json.load(file)
                             
wordcloud = []
for row in comment_objs:
    if row["predict"] == 2:
        continue
    ws = pseg.cut(row['content'])
    new_ws = []
    for word, flag in ws:
        if word in STOP_WORDS:
            continue
        if word not in STOP_WORDS and flag == 'n':
            wordcloud.append(word)


wordcloud_generator(" ".join(wordcloud), "C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-demo\\data-sources\\wordcloud.png")
wordcloud_generator(" ".join(wordcloud), "C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-management\\public\\data-sources\\wordcloud.png")

print("========== 產生關鍵字文字雲 結束 ==========")

print(f"========== 結束時間：{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} ==========")


print("========== 發送 Line 通知 開始 ==========")

if new_comments_cnt == 0:
    push_text = f"""已蒐集完畢，無新增評論數。"""
else:
    push_text = f"""已蒐集完畢，新增評論數：{new_comments_cnt}筆。\n正向評論數：{new_comments_predict['positive']}筆，中立評論數：{new_comments_predict['neutral']}筆，負向評論數：{new_comments_predict['negative']}筆。"""

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + channel_access_token
    }
    res = requests.post("https://api.line.me/v2/bot/message/push", json={"to": user_id, "messages": [{"type": "text", "text": push_text}]}, headers=headers)

print(push_text)
print("========== 發送 Line 通知 結束 ==========")