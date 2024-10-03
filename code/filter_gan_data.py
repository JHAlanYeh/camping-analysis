import pandas as pd
import json
import os
import numpy as np
from sklearn.utils import shuffle
import random
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


def random_masking(words, mask_token="[MASK]", mask_prob=0.15):
    num_words = len(words)
    num_to_mask = int(num_words * mask_prob)

    # 隨機選擇需要掩碼的詞
    mask_indices = random.sample(range(num_words), num_to_mask)
    masked_words = words.copy()

    for idx in mask_indices:
        masked_words[idx] = mask_token

    return "".join(masked_words)


#### filter gan data ####
# df = pd.read_csv("new_data/docs_0819/Final_Taide/taide_type1_merge_df.csv")

# gan_df = df[df["origin"] == 0]

# low_df = gan_df[gan_df["rating"] <= 2]
# mid_df = gan_df[gan_df["rating"] == 3]
# print(len(mid_df), len(low_df))


# conditions = [
#     gan_df['rating'] >= 4,
#     gan_df['rating'] == 3,
#     gan_df['rating'] <= 2,
# ]

# results = ["正向", "中立", "負向"]
# gan_df['label'] = np.select(conditions, results)
# gan_df.to_csv(f"new_data/docs_0819/Final_GPT4o_Mini/taide_type1_gan_df.csv", index=False, encoding="utf-8-sig")



gan_df = pd.read_csv(f"new_data/docs_0819/Final_TaiwanLLM/taiwanllm_type2_prompt_easy_gan_df.csv")
origin_train_df = pd.read_csv("new_data/docs_0819/Final_Origin/Type2_Result/origin_type2_train_df.csv")
origin_test_df = pd.read_csv("new_data/docs_0819/Final_Origin/Type2_Result/type2_test_df.csv")

test_mid_seq = list(dict.fromkeys(origin_test_df[origin_test_df["rating"] == 3]["sequence_num"].tolist()))
test_low_seq = list(dict.fromkeys(origin_test_df[origin_test_df["rating"] <= 2]["sequence_num"].tolist()))

# print(test_mid_seq)
# print(test_low_seq)

gan_mid_df = gan_df[gan_df["rating"] == 3]
gan_low_df = gan_df[gan_df["rating"] <= 2]
print(len(gan_mid_df), len(gan_low_df))


gan_test_mid_df = gan_mid_df[gan_mid_df["sequence_num"].isin(test_mid_seq)]
gan_test_low_df = gan_low_df[gan_low_df["sequence_num"].isin(test_low_seq)]
print("測試生成資料")
print(len(gan_test_mid_df), len(gan_test_low_df))

gan_train_mid_df = gan_mid_df[~gan_mid_df["sequence_num"].isin(test_mid_seq)]
gan_train_low_df = gan_low_df[~gan_low_df["sequence_num"].isin(test_low_seq)]
print("訓練生成資料")
print(len(gan_train_mid_df), len(gan_train_low_df))


high_df = origin_train_df[origin_train_df["rating"] >= 4]
mid_df = origin_train_df[origin_train_df["rating"] == 3]
low_df = origin_train_df[origin_train_df["rating"] <= 2]
print("原始資料")
print(len(high_df), len(mid_df), len(low_df))

new_train_df = shuffle(pd.concat([high_df, mid_df, gan_test_mid_df, gan_train_mid_df.sample(len(high_df) -len(mid_df) - len(gan_test_mid_df)), low_df, gan_test_low_df, gan_train_low_df.sample(len(high_df) -len(low_df) - len(gan_test_low_df))]))


texts = []
for row, origin in zip(new_train_df['content'],  new_train_df['origin']):
    ws = jieba.cut(row, cut_all=False)
    new_ws = []
    for word in ws:
        if word not in STOP_WORDS:
            new_ws.append(word)
    texts.append("".join(new_ws))

# print(texts)
new_train_df["text"] = texts
new_train_df["synonyms"] = ""
new_train_df = shuffle(new_train_df)

conditions = [
    new_train_df['rating'] >= 4,
    new_train_df['rating'] == 3,
    new_train_df['rating'] <= 2,
]

results = [2, 1, 0]
new_train_df['label'] = np.select(conditions, results)


print(len(new_train_df))
new_train_df.to_csv("new_data/docs_0819/Final_TaiwanLLM/Type2_Result/taiwanllm_type2_prompt_easy_train_df.csv", index=False, encoding="utf-8-sig")