import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import jieba
from dateutil.relativedelta import relativedelta
import json
import emoji
import shutil

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

# ****************************************************************************** #
root_path = os.path.join(os.getcwd(), "new_data")

target_directory = "summary_comments"
target_path = os.path.join(root_path, target_directory)

type1_comments = []
type2_comments = []

type1_content = []
type2_content = []


for file in os.listdir(target_path):
    if ".json" not in file:
        continue
    target_file_path = os.path.join(target_path, file)
    # print(target_file_path)
    tf = open(target_file_path, encoding="utf-8-sig")
    target_data = json.load(tf)
    tf.close()

    for d in target_data:
        emojis = emoji.emoji_list(d["content"])
        for e in emojis:
            d["content"] = d["content"].replace(e["emoji"], "")

        ch_list = re.compile('[\u4e00-\u9fff]+').findall(d["content"])
        if len(ch_list) == 0:
            continue
        # d["content"] = re.sub('[^\u4e00-\u9fa5]+', '', d["content"])
        # print(d["content"])

        if len(d["content"]) <= 20 or len(d["content"]) >= 512:
            continue
        if "打卡" in d["content"] or "送" in d["content"]:
            continue
        if d["type"] == 1 and ("木屋" in d["content"] or "民宿" in d["content"] or "套房" in d["content"]):
            continue
        if d["content"].strip() == "":
            continue
        if d["rating"] < 1 or d["rating"] > 5:
            continue
    
        ws = jieba.lcut(d["content"], cut_all=False)
        new_ws = []
        for word in ws:
            if word not in STOP_WORDS:
                new_ws.append(word)
        d['text'] =  "".join(new_ws)

        if d["type"] == 1:
            if d['content'] not in type1_content:
                type1_comments.append(d)
                type1_content.append(d['content'])
        elif d["type"] == 2:
            if d['content'] not in type2_content:
                type2_comments.append(d)
                type2_content.append(d['content'])

df1 = pd.json_normalize(type1_comments)
df2 = pd.json_normalize(type2_comments)

df1_3 = df1[df1["rating"] <= 3]
df2_3 = df2[df2["rating"] <= 3]
df1_5 = df1[(df1["rating"] > 3) & (df1["content"].str.len() > 20)]
df2_5 = df2[(df2["rating"] > 3) & (df2["content"].str.len() > 20)]

# print(len(df1_3), len(df1_5))
# print(len(df2_3), len(df2_5))

df_1_5_new= df1_5.sample(frac=0.41)
df_2_5_new = df2_5.sample(frac=0.29)

# print("=========================")
# print(len(df_1_5_new), len(df_2_5_new))

df1_final = pd.concat([df1_3, df_1_5_new]).sample(frac=1)
df2_final = pd.concat([df2_3, df_2_5_new]).sample(frac=1)

df1_final['sequence_num'] = df1_final.reset_index().index
df2_final['sequence_num'] = df2_final.reset_index().index

# 1/2星 = -1(負面), 3星 = 0(中等), 4/5星 = 1(正面)
#define conditions
conditions1 = [
    df1_final['rating'] >= 4,
    df1_final['rating'] == 3,
    df1_final['rating'] <= 2,
]

conditions2 = [
df2_final['rating'] >= 4,
df2_final['rating'] == 3,
df2_final['rating'] <= 2,
]

#define results
results = [1, 0, -1]
df1_final['status'] = np.select(conditions1, results)
df1_final['origin'] = 1
df2_final['status'] = np.select(conditions2, results)
df2_final['origin'] = 1

# print(df1.head(5))
# print(df2.head(5))

print("=========================")
print(len(df1_final))
print(len(df1_final[df1_final["status"] == -1]), len(df1_final[df1_final["status"] == 0]), len(df1_final[df1_final["status"] == 1]))
print(len(df2_final))
print(len(df2_final[df2_final["status"] == -1]), len(df2_final[df2_final["status"] == 0]), len(df2_final[df2_final["status"] == 1]))


df_path = os.path.join(root_path, "docs")

df1_final.to_csv(os.path.join(df_path, "type1_comments_0804.csv"), encoding="utf-8-sig", index=False)
df2_final.to_csv(os.path.join(df_path, "type2_comments_0804.csv"), encoding="utf-8-sig", index=False)
