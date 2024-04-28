from openai import OpenAI
import pandas as pd
import re
from opencc import OpenCC
import numpy as np
from sklearn.utils import shuffle

OPENAI_API_KEY="sk-proj-kSrjk2DSkQsvNBbGzzKeT3BlbkFJ6oBEEjuKHXj7W1LPFyju"
cn2zh = OpenCC('s2twp')
df = pd.read_csv("../docs/origin/type2_origin.csv", encoding="utf-8-sig")
gan_df = pd.read_csv('../docs/llmgan/type2_gan_df_3.csv')

gan_data = []
for index, row in list(gan_df.iterrows()):
    gan_data.append(dict(row))
print(len(gan_data))

client = OpenAI(
  api_key=OPENAI_API_KEY,
)

flag = False

p = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')

for index, row in df.iterrows():
    temp_data = []
    if len(row["content"]) >= 512:
        continue
    if row["rating"] >= 4:
        continue

    clean_content = re.sub(p,'', row["content"])
    if not clean_content.startswith("帳篷裡面環境糟糕") and flag == False:
        continue

    print("====================================")
    print(f"Origin: {clean_content}")

    print("\n增生文本如下：\n")
    while True:
 
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "user", "content": f"{clean_content} \n\n 以上評論請使用繁體中文來換句話說，盡量表達出句中所提到每一個重點，照著這個規則請列出類似的評論"},
            ],
            max_tokens=4053,
        )
      

        response = completion.choices[0].message.content
        response = cn2zh.convert(response)
        if response not in temp_data and response != clean_content:
            print(f"{len(temp_data) + 1}. {response}")
            temp_data.append(response)

        if len(temp_data) >= 5:
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

    flag = True


    gan_df = pd.json_normalize(gan_data)
    merge_df = pd.concat([df, gan_df])
    gan_df.to_csv('../docs/llmgan/type2_gan_df.csv', index=False, encoding="utf-8-sig")
    merge_df.to_csv('../docs/llmgan/type2_llm_gan_merge.csv', index=False, encoding="utf-8-sig")

    print("====================================")


# type2_df = pd.read_csv('../docs/llmgan/type2_llm_gan_merge.csv')
# type1_df = pd.read_csv('../docs/llmgan/type1_llm_gan_merge.csv')

# merge_df = pd.concat([type1_df, type2_df])
# merge_df = shuffle(merge_df).reset_index(drop=True) 
