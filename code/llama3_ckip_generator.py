# pip install transformers>=4.34
# pip install accelerate

from openai import OpenAI
import pandas as pd
import re
from opencc import OpenCC
import numpy as np
from sklearn.utils import shuffle
import torch
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

TOKEN = "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"
login(TOKEN)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = "../taide_Llama3_TAIDE_LX_8B_Chat_Alpha1_4bit/taide-8b-a.3-q4_k_m.gguf"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, torch_dtype="auto", device_map="auto", token=TOKEN
# )

flag = False
cn2zh = OpenCC('s2twp')
df = pd.read_csv("../docs/origin/type1_origin.csv", encoding="utf-8-sig")
# gan_df = pd.read_csv('../docs/llama3_ckip/type1_gan_df_13.csv')

gan_data = []
# for index, row in list(gan_df.iterrows()):
#     gan_data.append(dict(row))
# print(len(gan_data))

p = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')

for index, row in df.iterrows():
    temp_data = []
    if len(row["content"]) >= 512:
        continue
    if row["rating"] >= 4:
        continue

    clean_content = re.sub(p,'', row["content"])
    # if not clean_content.startswith("希望營區能管制小孩放") and flag == False:
    #     continue
   

    print("====================================")
    print(f"Origin: {clean_content}")

    print("\n增生文本如下：\n")
    while True:
        messages = [
            {
                "role": "user", 
                "content": f"{clean_content} \n\n  以上露營評論請使用繁體中文來換句話說，照著這個規則請生成評論"},
        ]
        
        response  = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )

        # outputs = model.generate(
        #     input_ids,
        #     max_new_tokens=8196,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.9,
        # )
        # embeddings = outputs[0][input_ids.shape[-1]:]
        # response = tokenizer.decode(embeddings, skip_special_tokens=True)
        print(response)
        response = cn2zh.convert(response)
        if response not in temp_data and response != clean_content:
            print(f"{len(temp_data) + 1}. {response}")
            temp_data.append(response)

        if len(temp_data) >= 3:
            break

    for t in temp_data:
        gan_data.append({
            "content": t,
            "rating": row["rating"],
            "type": row["type"],
            "status": row["status"],
            "tokenization": "",
            "publishedDate": row["publishedDate"],
            "origin": 0
        })

    # flag = True
    gan_df = pd.json_normalize(gan_data)
    merge_df = pd.concat([df, gan_df])
    gan_df.to_csv('../docs/llama3_ckip/type1_gan_df.csv', index=False, encoding="utf-8-sig")
    merge_df.to_csv('../docs/llama3_ckip/type1_llm_gan_merge.csv', index=False, encoding="utf-8-sig")
