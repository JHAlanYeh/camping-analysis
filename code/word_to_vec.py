# coding:utf-8  
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties as font
import jieba
import jieba.posseg as pseg
jieba.load_userdict('code\\custom_dict.txt')
jieba.set_dictionary('code\\dict.txt.big')


tw_font = font(fname="NotoSansTC-VariableFont_wght.ttf")

f = open('code\\stopwords_zh_TW.dat.txt', encoding="utf-8")
STOP_WORDS = []
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))

f = open('code\\stopwords.txt', encoding="utf-8")
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))

def wordcloud_generator(words, file_name):
    #文字雲造型圖片
    # mask = np.array(Image.open('picture.png')) #文字雲形狀
    # 從 Google 下載的中文字型
    font = 'SourceHanSansTW-Regular.otf'
    #背景顏色預設黑色，改為白色、使用指定圖形、使用指定字體
    my_wordcloud = WordCloud(width=600, height=400,background_color='white', font_path=font).generate(words)
    plt.imshow(my_wordcloud)
    plt.axis("off")
    # plt.show()
    #存檔
    my_wordcloud.to_file(file_name)

def countWord(text):
    counts={}
    for word in text: 
        if len(word) == 1 or word=='\n':#单个词和换行符不计算在内
            continue
        else:
            if word not in counts.keys():
                counts[word]=1
            else:
                counts[word]+=1
    return counts

def drawBar(countdict,RANGE, heng):
    #函数来源于：https://blog.csdn.net/leokingszx/article/details/101456624，有改动
    #dicdata：字典的数据。
    #RANGE：截取显示的字典的长度。
    #heng=0，代表条状图的柱子是竖直向上的。heng=1，代表柱子是横向的。考虑到文字是从左到右的，让柱子横向排列更容易观察坐标轴。
    by_value = sorted(countdict.items(),key = lambda item:item[1],reverse=True)
    print(by_value[:20])
    x = []
    y = []
    plt.figure(figsize=(9, 6))
    plt.yticks(font=tw_font, fontsize=10)
    plt.xticks(font=tw_font, fontsize=10)
    for d in by_value:
        x.append(d[0])
        y.append(d[1])
    if heng == 0:
        plt.bar(x[0:RANGE], y[0:RANGE])
        plt.savefig("new_data/docs_0819/Final_Origin/type2_wordcount_bar.png")
        return 
    elif heng == 1:
        plt.barh(x[0:RANGE], y[0:RANGE])
        plt.savefig("new_data/docs_0819/Final_Origin/type2_wordcount_hor.png")
        return 
    else:
        return "heng的值仅为0或1！"


origin_df = pd.read_csv("new_data/docs_0819/Final_Origin/type2_comments_origin.csv")

corpus = []
wordcloud = []
noun_freq = {}
for row in origin_df['content']:
    ws = pseg.cut(row)
    new_ws = []
    for word, flag in ws:
        if word in STOP_WORDS:
            continue
        if word not in STOP_WORDS and flag == 'n':
            new_ws.append(word)
            wordcloud.append(word)
        if word in noun_freq:
            noun_freq[word] += 1
        else:
            noun_freq[word] = 1
    corpus.append(" ".join(new_ws))

# 計算總詞數
total_nouns = sum(noun_freq.values())

# 計算每個名詞的比重
noun_percentage = {word: freq / total_nouns for word, freq in noun_freq.items()}
word_percentage = sorted(noun_percentage.items(), key=lambda x: x[1], reverse=True)
print(word_percentage[0:10])


# # ---------------------------------------------------- 
# vectorizer = CountVectorizer(stop_words=None)  
# #计算个词语出现的次数  
# X = vectorizer.fit_transform([" ".join(wordcloud)])  
# #获取词袋中所有文本关键词  
# word = vectorizer.get_feature_names_out()  
# #查看词频结果  
# df_word =  pd.DataFrame(X.toarray(),columns=word)


# #类调用  
# transformer = TfidfTransformer(smooth_idf=True,norm='l2',use_idf=True)  
# print(transformer)
# #将计算好的词频矩阵X统计成TF-IDF值  
# tfidf = transformer.fit_transform(X)  
# #查看计算的tf-idf
# df_word_tfidf = pd.DataFrame(tfidf.toarray(),columns=word)
# #查看计算的idf
# df_word_idf = pd.DataFrame(list(zip(word,transformer.idf_)),columns=['单词','idf'])
# df_word_idf = df_word_idf.sort_values(by=['idf'], ascending=True)
# print(df_word_idf[0:10])

# # ---------------------------------------------------- 

wordcloud_generator(" ".join(wordcloud), "new_data/docs_0819/Final_Origin/type2_wordcloud.png")
countdict=countWord(wordcloud)#生成词频字典
drawBar(countdict,15,0)#绘制词语出现次数前10的竖向条形图 
drawBar(countdict,15,1)#绘制词语出现次数前20的横向条形图     