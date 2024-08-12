import pandas as pd
from sklearn.utils import shuffle
import jieba
import numpy as np
import random
from BackTranslation import BackTranslation

trans = BackTranslation(url=[
      'translate.google.com',
      'translate.google.co.kr',
    ], proxies={'http': '127.0.0.1:1235', 'http://host.name': '127.0.0.1:4013'})


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

# ****************************************************************************** #


df = pd.read_csv("new_data/docs_0804/Final_GPT4o/gpt4o_type1_merge_train_df_2_20240812_3.csv", encoding="utf-8-sig")

texts = []
synonyms = []
for index, row in df.iterrows():
    if type(row['synonyms']) == str:
        continue
    if row['origin'] == 0 and np.isnan(row['synonyms']):
        print(row['origin'], row['synonyms'])
        result = trans.translate(row['content'], src='zh-tw', tmp = 'en')
        translate_text = result.result_text
        print(translate_text)
        df.loc[index, 'synonyms'] = translate_text
        cut_content = translate_text
    else:
        df.loc[index, 'synonyms'] = row['content']
        cut_content =  row['content']


    ws = jieba.cut(cut_content, cut_all=False)
    new_ws = []
    for word in ws:
        if word not in STOP_WORDS:
            new_ws.append(word)
    mask_text = random_masking(new_ws)
    df.loc[index, 'text'] = mask_text

    df.to_csv('new_data/docs_0804/Final_GPT4o/gpt4o_type1_merge_train_df_2_20240812_3.csv', index=False, encoding="utf-8-sig")


# type2_mid_gan_df = pd.read_csv("new_data/docs_0804/llama3_type2_mid_gan_dataset.csv", encoding="utf-8-sig")
# type2_low_gan_df = pd.read_csv("new_data/docs_0804/llama3_type2_low_gan_dataset.csv", encoding="utf-8-sig")
# type2_origin_df = pd.read_csv("new_data/docs_0804/type2_train_df.csv", encoding="utf-8-sig")

# type2_positive = len(type2_origin_df[type2_origin_df["rating"] >= 4])
# type2_negative = len(type2_origin_df[type2_origin_df["rating"] <= 2]) + len(type2_low_gan_df[type2_low_gan_df["rating"] <= 2])
# type2_mid = len(type2_origin_df[type2_origin_df["rating"] == 3]) + len(type2_mid_gan_df[type2_mid_gan_df["rating"] == 3])
# print(type2_positive, type2_negative, type2_mid)

# if type2_negative > type2_positive:
#     type2_low_gan_df = type2_low_gan_df.sample(type2_positive - len(type2_origin_df[type2_origin_df["rating"] <= 2]))

# if type2_mid > type2_positive:
#     type2_mid_gan_df = type2_mid_gan_df.sample(type2_positive - len(type2_origin_df[type2_origin_df["rating"] == 3]))

# type2_positive = len(type2_origin_df[type2_origin_df["rating"] >= 4])
# type2_negative = len(type2_origin_df[type2_origin_df["rating"] <= 2]) + len(type2_low_gan_df[type2_low_gan_df["rating"] <= 2])
# type2_mid = len(type2_origin_df[type2_origin_df["rating"] == 3]) + len(type2_mid_gan_df[type2_mid_gan_df["rating"] == 3])
# print(type2_positive, type2_negative, type2_mid)

# type2_merge_llama3_df = pd.concat([type2_mid_gan_df, type2_low_gan_df, type2_origin_df])
# type2_merge_llama3_df.to_csv('new_data/docs_0804/Final_Llama3/llama3_type2_merge_train_dataset.csv', index=False, encoding="utf-8-sig")
