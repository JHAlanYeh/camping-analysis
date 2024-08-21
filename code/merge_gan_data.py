import pandas as pd
from sklearn.utils import shuffle
import numpy as np
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

# ****************************************************************************** #


mid_gan_df = pd.read_csv("new_data/docs_0819/Final_GPT4o_Mini/gpt4o_type2_mid_gan_df.csv", encoding="utf-8-sig")
low_gan_df = pd.read_csv("new_data/docs_0819/Final_GPT4o_Mini/gpt4o_type2_low_gan_df.csv", encoding="utf-8-sig")


mid_conditions = [
    mid_gan_df['status'] == -1,
    mid_gan_df['status'] == 0,
    mid_gan_df['status'] == 1,
]

low_conditions = [
    low_gan_df['status'] == -1,
    low_gan_df['status'] == 0,
    low_gan_df['status'] == 1,
]

# create a list of the values we want to assign for each condition
values = [0, 1, 2]

# create a new column and use np.select to assign values to it using our lists as arguments
mid_gan_df['label'] = np.select(mid_conditions, values)
low_gan_df['label'] = np.select(low_conditions, values)


low_gan_df = low_gan_df[['content', 'rating', 'status', 'type', 'label', 'sequence_num', 'publishedDate', 'origin']]
mid_gan_df = mid_gan_df[['content', 'rating', 'status', 'type', 'label', 'sequence_num', 'publishedDate', 'origin']]

print(low_gan_df.columns)
print(mid_gan_df.columns)

merge_df = pd.concat([low_gan_df, mid_gan_df])

texts = []
synonyms = []
for row, origin in zip(merge_df['content'],  merge_df['origin']):
    ws = jieba.cut(row, cut_all=False)
    new_ws = []
    for word in ws:
        if word not in STOP_WORDS:
            new_ws.append(word)
    texts.append("".join(new_ws))

merge_df["text"] = texts
merge_df = shuffle(merge_df)
merge_df.to_csv('new_data/docs_0819/Final_GPT4o_Mini/gpt4o_type2_gan_df.csv', index=False, encoding="utf-8-sig")
