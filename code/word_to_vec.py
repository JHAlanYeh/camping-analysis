# coding:utf-8  
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer 

import jieba
import jieba.posseg as pseg
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


origin_df = pd.read_csv("new_data/docs_0819/Final_Origin/type1_comments_origin.csv")

corpus = []
noun_freq = {}
for row in origin_df['content']:
    ws = pseg.cut(row)
    new_ws = []
    for word, flag in ws:
        if word in STOP_WORDS:
            continue
        if word not in STOP_WORDS and flag == 'n':
            new_ws.append(word)
        if word in noun_freq:
            noun_freq[word] += 1
        else:
            noun_freq[word] = 1
    corpus.append(" ".join(new_ws))

# type2_df = pd.read_csv("new_data/docs_0819/Final_Origin/type2_comments_origin.csv")
# 計算總詞數
total_nouns = sum(noun_freq.values())

# 計算每個名詞的比重
noun_percentage = {word: freq / total_nouns for word, freq in noun_freq.items()}
word_percentage = sorted(noun_percentage.items(), key=lambda x: x[1], reverse=True)
print(word_percentage[0:10])




# #将文本中的词语转换为词频矩阵  
# vectorizer = CountVectorizer(stop_words=None)  
# #计算个词语出现的次数  
# X = vectorizer.fit_transform(corpus) 
# #获取词袋中所有文本关键词  
# word = vectorizer.get_feature_names_out()  
# # 計算每個詞在多少個文檔中出現（非零值的文檔數）
# doc_count = (X > 0).sum(axis=0).A1  # A1 表示轉為一維陣列

# # 語料庫中的文檔總數
# total_documents = len(corpus)
# # 計算每個詞的 TF（詞頻）
# tf_matrix = X.toarray()

# # 計算每個詞在整個語料中的詞彙頻率
# word_freq_in_corpus = np.sum(tf_matrix, axis=0)

# # 計算 IWF，IWF = log(詞彙總數 / 該詞出現的頻數)
# iwf = np.log(len(word) / word_freq_in_corpus)

# # 輸出每個詞的 IWF 分數
# # for term, score in zip(word, iwf):
# #     print(f"Word: {term}, IWF: {score}")

# # 計算每個詞的 TF-IWF
# # tf_iwf_matrix = tf_matrix * iwf
# df_word_iwf = pd.DataFrame(list(zip(word, iwf)),columns=['單詞','tf-iwf'])
# df_word_iwf = df_word_iwf.sort_values(by=['tf-iwf'], ascending=False)

# # print(df_word_iwf[0:10])


from sklearn.feature_extraction.text import TfidfVectorizer

# 定義文件集
documents = [
    "這是一個測試文件 這是另一個文件 這個文件是用來測試 TF-IDF 的"
]

# 創建 TfidfVectorizer 物件
vectorizer = TfidfVectorizer()

# 計算 TF-IDF
tfidf_matrix = vectorizer.fit_transform(documents)

# 獲取詞彙表
feature_names = vectorizer.get_feature_names_out()

# 顯示 TF-IDF 結果
for doc_index, doc in enumerate(tfidf_matrix.toarray()):
    print(f"文件 {doc_index + 1}:")
    for word_index, score in enumerate(doc):
        if score > 0:
            print(f"詞: {feature_names[word_index]}, TF-IDF: {score}")


