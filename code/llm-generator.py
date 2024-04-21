from openai import OpenAI
import os
import pandas as pd
import re
from opencc import OpenCC

OPENAI_API_KEY="sk-proj-nIPB3DWthajrswj4YG5kT3BlbkFJ0CuuXYlB2UkO967o0bSA"
cn2zh = OpenCC('s2twp')
df = pd.read_csv("../docs/origin/type_origin.csv")
gan_data = []




# response = cn2zh.convert("新營區，強烈建議包場，因為場地不大。熱水很棒，但蓮蓬頭的角度稍微有點微妙。包場靠近廁所的位置草皮很不錯，可惜我們沒有享受到，因為訂了比較邊緣的位置。規劃還不夠成熟，明顯還在建造，我們的位置比較偏僻，草皮還在長，垃圾桶也不明顯，需要自己搬空籃子，洗手台也在邊緣，沒有廚餘桶，得使用其他區的，但裡面都是垃圾！電源也很特別，要每四小時去按一次充電柱，幸好不是夏天，不然應該會很熱。我們正好碰上隔壁民宿辦趴，整個山谷都是歌聲，一直唱到深夜，然後還有重機在山谷飆車，早上又有狗吠雞叫，是相當特別的體驗。老闆非常熱情且客氣，會教如何炒咖啡豆，分享附近景點，還有賞星空。")
# print(response)


client = OpenAI(
  api_key=OPENAI_API_KEY,
)

p = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')

for index, row in df.iterrows():
    temp_data = []
    if len(row["content"]) >= 512:
        continue
    if row["rating"] >= 4:
        continue

    clean_content = re.sub(p,'', row["content"]) 
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


    gan_df = pd.json_normalize(gan_data)
    merge_df = pd.concat([df, gan_df])
    gan_df.to_csv('../docs/llmgan/gan_df.csv', index=False, encoding="utf-8-sig")
    merge_df.to_csv('../docs/llmgan/llm_gan_merge.csv', index=False, encoding="utf-8-sig")

    print("====================================")