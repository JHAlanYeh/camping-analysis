import jieba
import pandas as pd


jieba.load_userdict('custom_dict.txt')

f = open('stopwords_zh_TW.dat.txt', encoding="utf-8")
STOP_WORDS = []
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))

f = open('stopwords.txt', encoding="utf-8")
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))


df = pd.read_csv("../docs/gan/all_gan_merge.csv")

for index, row in df.iterrows():
    if str(row['tokenization']).strip() == "nan":
        ws = jieba.lcut(row['content'], cut_all=False)
        new_ws = []
        for word in ws:
            if word not in STOP_WORDS:
                new_ws.append(word)
        # print(" | ".join(new_ws))
        df.at[index,'tokenization'] = " | ".join(new_ws)


pd.DataFrame(df).to_csv("../docs/gan/all_gan_merge_tokenization.csv", index=False, encoding='utf-8-sig')