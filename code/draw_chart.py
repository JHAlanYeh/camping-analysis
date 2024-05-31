import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as font
import matplotlib


tw_font = font(fname="NotoSansTC-VariableFont_wght.ttf")

def addlabels(x,y, padding = 0.07):
  for i in range(len(x)):
    plt.text(i-padding, y[i]+20, y[i])

LABELS = ['負向', '中立', '正向']
STAR_LABELS = ['一星', '二星', '三星', '四星', '五星']

def type1_origin_chart():
    df = pd.read_csv('new_data/docs/type1_comments_low_v2.csv')

    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(傳統露營)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs/type1_origin_v2.png')


def type1_aug_star_chart():
    df = pd.read_csv('../docs/llama3gan/type1_llm_gan_merge.csv')
    values = []

    values.append(len(df[df["rating"] == 1]))
    values.append(len(df[df["rating"] == 2]))
    values.append(len(df[df["rating"] == 3]))
    values.append(len(df[df["rating"] == 4]))
    values.append(len(df[df["rating"] == 5]))

    ser = pd.Series(values, index=STAR_LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=STAR_LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('增生後星等分布(傳統露營)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(STAR_LABELS, values, 0.15)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('../docs/llama3gan/type1_aug_star.png')

def type1_aug_chart():
    df = pd.read_csv('../docs/llama3gan/type1_llm_gan_merge.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('增生後資料(傳統露營)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('../docs/llama3gan/type1_aug.png')

def type2_origin_chart():
    df = pd.read_csv('new_data/docs/type2_comments_low_v2.csv')

    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(懶人露營)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs/type2_origin_v2.png')

def type2_aug_star_chart():
    df = pd.read_csv('../docs/llama3gan/type2_llm_gan_merge.csv')
    values = []

    values.append(len(df[df["rating"] == 1]))
    values.append(len(df[df["rating"] == 2]))
    values.append(len(df[df["rating"] == 3]))
    values.append(len(df[df["rating"] == 4]))
    values.append(len(df[df["rating"] == 5]))

    ser = pd.Series(values, index=STAR_LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=STAR_LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('增生後星等分布(懶人露營)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(STAR_LABELS, values, 0.15)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('../docs/llama3gan/type2_aug_star.png')


def type2_aug_chart():
    df = pd.read_csv('../docs/llama3gan/type2_llm_gan_merge.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type2_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type2_merge_df.plot(kind='bar')

    plt.title('增生後資料(懶人露營)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('../docs/llama3gan/type2_aug.png')


if __name__ == "__main__":
    # type1_aug_chart()
    # type1_aug_star_chart()
    # type2_aug_chart()
    # type2_aug_star_chart()

    type1_origin_chart()
    type2_origin_chart()