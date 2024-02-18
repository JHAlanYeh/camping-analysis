import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import json
from zhconv import convert

def convert2zhTW():
    new_word_arr = []
    f = open('stopwords.dat.txt', encoding="utf-8-sig")
    for line in f.readlines():
        new_word_arr.append(convert(line, 'zh-hant'))
    f.close

    f = open('stopwords_zh_TW.dat.txt', 'w', encoding="utf-8-sig")
    for nw in new_word_arr:
        f.write(nw)
    f.close()

def get_tokenization(camping_type):
    f = open('stopwords_zh_TW.dat.txt', encoding="utf-8")
    STOP_WORDS = []
    lines = f.readlines()
    for line in lines:
        STOP_WORDS.append(line.rstrip('\n'))

    tokenization_arr = []

    dir_path = os.path.join(os.getcwd(), "data")
    asiayo_dir = os.path.join(dir_path, "asiayo_comments")
    easycamp_dir = os.path.join(dir_path, "easycamp_comments")
    klook_dir = os.path.join(dir_path, "klook_comments")

    asiayo_type_dir = os.path.join(asiayo_dir, camping_type)
    if os.path.isdir(asiayo_type_dir):
        for file in os.listdir(asiayo_type_dir):
            file_path = os.path.join(asiayo_type_dir, file)
            f = open(file_path, encoding="utf-8-sig")
            data = json.load(f)
            for c in data["comments"]:
                tokenization_arr.append(c["tokenization"])
    
    easycamp_type_dir = os.path.join(easycamp_dir, camping_type)
    if os.path.isdir(easycamp_type_dir):
        for file in os.listdir(easycamp_type_dir):
            file_path = os.path.join(easycamp_type_dir, file)
            f = open(file_path, encoding="utf-8-sig")
            data = json.load(f)
            for c in data["comments"]:
                tokenization_arr.append(c["tokenization"])

    klook_type_dir = os.path.join(klook_dir, camping_type)
    if os.path.isdir(klook_type_dir):
        for file in os.listdir(klook_type_dir):
            file_path = os.path.join(klook_type_dir, file)
            f = open(file_path, encoding="utf-8-sig")
            data = json.load(f)
            for c in data["comments"]:
                tokenization_arr.append(c["tokenization"])

    return " ".join(tokenization_arr)


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


if __name__ == "__main__":
    # convert2zhTW()
    words_1 = get_tokenization("1")
    words_2 = get_tokenization("2")
    wordcloud_generator(words_1, "camping_type_1.png")
    wordcloud_generator(words_2, "camping_type_2.png")