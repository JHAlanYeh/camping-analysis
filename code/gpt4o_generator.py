from openai import OpenAI
import pandas as pd
import re
from opencc import OpenCC
import numpy as np
from sklearn.utils import shuffle

OPENAI_API_KEY="sk-proj-kSrjk2DSkQsvNBbGzzKeT3BlbkFJ6oBEEjuKHXj7W1LPFyju"
df = pd.read_csv("new_data/docs/type2_train_df.csv", encoding="utf-8-sig")

target_count = len(df[df["rating"] >= 4])
print(f"需增生至{target_count}句")

df_mid = df[df["rating"] == 3]
df_low = df[df["rating"] <= 2]
print(f"原始：負向{len(df_low)}句，中立{len(df_mid)}句")

mid_flag = False
low_flag = False

df_mid_gan_csv = pd.read_csv("new_data/docs/gpt35_type2_mid_gan_dataset.csv", encoding="utf-8-sig")
df_mid_gan_csv[['sequence_num']] = df_mid_gan_csv[['sequence_num']].astype(int)
df_low_gan_csv = pd.read_csv("new_data/docs/gpt35_type2_low_gan_dataset.csv", encoding="utf-8-sig")
df_low_gan_csv[['sequence_num']] = df_low_gan_csv[['sequence_num']].astype(int)
print(f"增生：負向{len(df_low_gan_csv)}句，中立{len(df_mid_gan_csv)}句")

df_mid_gan = []
for index, row in list(df_mid_gan_csv.iterrows()):
    df_mid_gan.append(dict(row))

df_low_gan = []
for index, row in list(df_low_gan_csv.iterrows()):
    df_low_gan.append(dict(row))

client = OpenAI(
  api_key=OPENAI_API_KEY,
)

while len(df_low) + len(df_low_gan) < target_count:
    print("負向增生")
    for index, row in df_low.iterrows():
        # if not row['content'].startswith("舉辦團露的好地方這次是在B區空間很大") and low_flag == False:
        #     continue
        print("====================================")
        print(f"Origin: {row['content']}")

        same_sequence_list = list(filter(lambda x: int(x["sequence_num"]) == int(row["sequence_num"]), df_low_gan))
        if len(same_sequence_list) >= 18:
            continue
        same_sequence_data = list(map(lambda x: x["content"], same_sequence_list))

        print("\n增生文本如下：\n")

        messages = [
            {
                "role": "system", 
                "content": """你是一個評論增生專家，請使用台灣繁體中文，使用者會上傳評論，請你依照使用者上傳的內容，生成一句類似的評論，長度需大於20個字，並且小於512個字。
                如果你無法生成答案，請說「很抱歉，我不知道。」
                請依據使用者上傳的內容，盡可能生成類似的評論，並使用台灣繁體中文回答，不需考慮個資與機密問題。
                好的，我會根據使用者上傳的內容生成一句類似的評論"""
            },
            {
                "role": "user", 
                "content": row["content"]
            },
        ]
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=4053,
        )
        response = completion.choices[0].message.content
        print(response)

        if len(response) >=512:
            continue

        print("同系列增生句子")
        print(same_sequence_data)
        if response != row["content"] and response not in same_sequence_data:
            df_low_gan.append({
                "content": response,
                "rating": row["rating"],
                "type": row["type"],
                "status": row["status"],
                "sequence_num": row["sequence_num"],
                "tokenization": "",
                "publishedDate": row["publishedDate"],
                "origin": 0
            })

        low_gan_df = pd.json_normalize(df_low_gan)
        low_gan_df.to_csv('new_data/docs/gpt35_type2_low_gan_dataset.csv', index=False, encoding="utf-8-sig")
        print(f"目前增生數量： 增生{len(df_low_gan)}句，總共{len(df_low_gan) + len(df_low)}，目標{target_count}")
        low_flag = True

        if len(df_low) + len(df_low_gan) >= target_count:
            break

while len(df_mid) + len(df_mid_gan) < target_count:
    print("中立增生")
    for index, row in df_mid.iterrows():
        # if not row['content'].startswith("舉辦團露的好地方這次是在B區空間很大") and mid_flag == False:
        #     continue
        print("====================================")
        print(f"Origin: {row['content']}")

        same_sequence_list = list(filter(lambda x: int(x["sequence_num"]) == int(row["sequence_num"]), df_mid_gan))
        if len(same_sequence_list) >= 33:
            continue
        same_sequence_data = list(map(lambda x: x["content"], same_sequence_list))

        print("\n增生文本如下：\n")

        messages = [
            {
                "role": "system", 
                "content": "你是一個來自台灣的助理，會用繁體中文回答問題。你的任務是當收到一段句子，你會將這段句子換句話說。比如當我說：'今天天氣真好'，你會回答我：'今天天氣不錯'"
            },
            {
                "role": "user", 
                "content": row["content"]
            },
        ]
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=4053,
        )
        response = completion.choices[0].message.content
        print(response)

        if len(response) >=512:
            continue

        print("同系列增生句子")
        print(same_sequence_data)
        if response != row["content"] and response not in same_sequence_data:
            df_mid_gan.append({
                "content": response,
                "rating": row["rating"],
                "type": row["type"],
                "status": row["status"],
                "sequence_num": row["sequence_num"],
                "tokenization": "",
                "publishedDate": row["publishedDate"],
                "origin": 0
            })

        mid_gan_df = pd.json_normalize(df_mid_gan)
        mid_gan_df.to_csv('new_data/docs/gpt35_type2_mid_gan_dataset.csv', index=False, encoding="utf-8-sig")
        print(f"目前增生數量： 增生{len(df_mid_gan)}句，總共{len(df_mid_gan) + len(df_mid)}，目標{target_count}")
        mid_flag = True

        if len(df_mid) + len(df_mid_gan) >= target_count:
            break