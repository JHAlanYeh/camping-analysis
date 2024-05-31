# pip install transformers>=4.34
# pip install accelerate

from openai import OpenAI
import pandas as pd
import re
from opencc import OpenCC
import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

TOKEN = "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"
login(TOKEN)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = "llm_model\Llama3-TAIDE-LX-8B-Chat-Alpha1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)


df = pd.read_csv("new_data/docs/type2_comments_low_v1.csv", encoding="utf-8-sig")
target_count = len(df[df["rating"] >= 4])
print(f"需增生至{target_count}句")

df_mid = df[df["rating"] == 3]
df_low = df[df["rating"] <= 2]
print(f"原始：負向{len(df_low)}句，中立{len(df_mid)}句")

mid_flag = False
low_flag = False

df_mid_gan_csv = pd.read_csv("new_data/docs/llama3_type2_mid_gan_df.csv", encoding="utf-8-sig")
df_mid_gan_csv[['sequence_num']] = df_mid_gan_csv[['sequence_num']].astype(int)
df_low_gan_csv = pd.read_csv("new_data/docs/llama3_type2_low_gan_df.csv", encoding="utf-8-sig")
df_low_gan_csv[['sequence_num']] = df_low_gan_csv[['sequence_num']].astype(int)
print(f"增生：負向{len(df_low_gan_csv)}句，中立{len(df_mid_gan_csv)}句")

df_mid_gan = []
for index, row in list(df_mid_gan_csv.iterrows()):
    df_mid_gan.append(dict(row))

df_low_gan = []
for index, row in list(df_low_gan_csv.iterrows()):
    df_low_gan.append(dict(row))

while len(df_low) + len(df_low_gan) < target_count:
    print("負向增生")
    for index, row in df_low.iterrows():
        # if not row['content'].startswith("1.場地很美，草皮很漂亮。2.帳帳相連，很擁擠，") and low_flag == False:
        #     continue
        print("====================================")
        print(f"Origin: {row['content']}")

        same_sequence_list = list(filter(lambda x: int(x["sequence_num"]) == int(row["sequence_num"]), df_low_gan))
        if len(same_sequence_list) >= 19:
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
        low_gan_df.to_csv('new_data/docs/llama3_type2_low_gan_df.csv', index=False, encoding="utf-8-sig")
        print(f"目前增生數量： 增生{len(df_low_gan)}句，總共{len(df_low_gan) + len(df_low)}，目標{target_count}")
        low_flag = True

        if len(df_low) + len(df_low_gan) >= target_count:
            break

# ========================================================= #
# 中立
while len(df_mid) + len(df_mid_gan) < target_count:
    print("中立增生")
    for index, row in df_mid.iterrows():
        # if not row['content'].startswith("1.營主很熱心，這麼熱的天氣，送來透心涼的點心") and mid_flag == False:
        #     continue
        print("====================================")
        print(f"Origin: {row['content']}")

        same_sequence_list = list(filter(lambda x: int(x["sequence_num"]) == int(row["sequence_num"]), df_mid_gan))
        if len(same_sequence_list) >= 19:
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
            }
        ]
        
        input_ids = tokenizer.apply_chat_template(
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
        mid_gan_df.to_csv('new_data/docs/llama3_type2_mid_gan_df.csv', index=False, encoding="utf-8-sig")
        print(f"目前增生數量： 增生{len(df_mid_gan)}句，總共{len(df_mid_gan) + len(df_mid)}，目標{target_count}")
        mid_flag = True

        if len(df_mid) + len(df_mid_gan) >= target_count:
            break