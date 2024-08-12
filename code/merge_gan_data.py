import pandas as pd
from sklearn.utils import shuffle
import jieba
import numpy as np

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


# type1_mid_gan_df = pd.read_csv("new_data/docs_0804/gpt4o_type1_mid_gan_train_df_3.csv", encoding="utf-8-sig")
type1_low_gan_df = pd.read_csv("new_data/docs_0804/gpt4o_type1_low_gan_train_df_2.csv", encoding="utf-8-sig")
type1_origin_df = pd.read_csv("new_data/docs_0804/Final_Origin/Type1_Result/train_df_2.csv", encoding="utf-8-sig")

# mid_conditions = [
#     type1_mid_gan_df['status'] == -1,
#     type1_mid_gan_df['status'] == 0,
#     type1_mid_gan_df['status'] == 1,
# ]

low_conditions = [
    type1_low_gan_df['status'] == -1,
    type1_low_gan_df['status'] == 0,
    type1_low_gan_df['status'] == 1,
]

# create a list of the values we want to assign for each condition
values = [0, 1, 2]

# create a new column and use np.select to assign values to it using our lists as arguments
# type1_mid_gan_df['label'] = np.select(mid_conditions, values)
type1_low_gan_df['label'] = np.select(low_conditions, values)


type1_low_gan_df = type1_low_gan_df[['content', 'rating', 'status', 'type', 'label', 'sequence_num', 'publishedDate', 'origin']]
# type1_mid_gan_df = type1_mid_gan_df[['content', 'rating', 'status', 'type', 'label', 'sequence_num', 'publishedDate', 'origin']]
type1_origin_df['origin'] = 1
type1_origin_df = type1_origin_df[['content', 'rating', 'status', 'type', 'label', 'sequence_num', 'publishedDate', 'origin']]

print(type1_origin_df.columns)
print(type1_low_gan_df.columns)
# print(type1_mid_gan_df.columns)

type1_merge_df = pd.concat([type1_origin_df, type1_low_gan_df])

type1_positive = len(type1_merge_df[type1_merge_df["rating"] >= 4])
type1_negative = len(type1_merge_df[type1_merge_df["rating"] <= 3])
# type1_mid = len(type1_merge_df[type1_merge_df["rating"] <= 3])
print(type1_positive, type1_negative)

texts = []
for row in type1_merge_df['content']:
    ws = jieba.lcut(row, cut_all=False)
    new_ws = []
    for word in ws:
        if word not in STOP_WORDS:
            new_ws.append(word)
    texts.append("".join(new_ws))

type1_merge_df["text"] = texts
type1_merge_df = shuffle(type1_merge_df)

type1_merge_df.to_csv('new_data/docs_0804/Final_GPT4o/gpt4o_type1_merge_train_df_2_20240812.csv', index=False, encoding="utf-8-sig")


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

