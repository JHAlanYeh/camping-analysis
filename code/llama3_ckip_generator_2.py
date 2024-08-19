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
import jieba
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_id = "llm_model\Llama3-TAIDE-LX-8B-Chat-Alpha1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto", quantization_config=nf4_config
)


df = pd.read_csv("new_data\\docs_0804\\Final_Origin\\Type1_Result\\train_df_2.csv", encoding="utf-8-sig")
target_count = len(df[df["rating"] >= 4])
print(f"需增生至{target_count}句")

df_high = df[df["rating"] > 3]
df_low = df[df["rating"] <= 3]
print(f"原始：負向{len(df_low)}句")

low_flag = False

df_low_gan_csv = pd.read_csv("new_data/docs_0804/taide_type1_low_gan_train_df_2.csv", encoding="utf-8-sig")
df_low_gan_csv[['sequence_num']] = df_low_gan_csv[['sequence_num']].astype(int)
print(f"增生：負向{len(df_low_gan_csv)}句")

df_low_gan = []
for index, row in list(df_low_gan_csv.iterrows()):
    df_low_gan.append(dict(row))

while len(df_low) + len(df_low_gan) < target_count:
    print("負向增生")
    for index, row in df_low.iterrows():
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
                "content": """您是一個資料增生的專家，我會上傳露營的評論，請您依照上傳的內容，並且使用台灣繁體中文，盡可能生成類似的露營評論，生成的句子長度需大於20個字，並且小於512個字，生成的句子請直接回答，不再多加任何贅字。上傳的評論大多為負向或中立的情緒，請照著相同的情緒生成類似的句子。如果您無法生成類似的評論，一律請說「非常抱歉，我無法生成該露營評論的相似內容。」，請不要回覆我上傳的評論。"""
                # "content": """您是一個資料增生的專家，我會上傳露營的評論，請您依照上傳的內容，並且使用台灣繁體中文，盡可能生成這個露營評論的負向情緒評論，生成的句子長度需大於20個字，並且小於512個字，生成的句子請直接回答，不再多加任何贅字。如果您無法生成類似的評論，一律請說「非常抱歉，我無法生成類似該露營評論的負向情緒內容。」"""
            },
            {
                "role": "assistant", 
                "content": "好的，請您上傳露營的評論內容。"
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
        if "抱歉" in response and "無法生成" in response and "相似內容" in response:
            continue
        if "非常抱歉，我無法生成該露營評論的相似內容。" in response:
            continue

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
        low_gan_df.to_csv('new_data/docs_0804/taide_type1_low_gan_train_df_2.csv', index=False, encoding="utf-8-sig")
        print(f"目前增生數量： 增生{len(df_low_gan)}句，總共{len(df_low_gan) + len(df_low)}，目標{target_count}")
        low_flag = True

        if len(df_low) + len(df_low_gan) >= target_count:
            break

