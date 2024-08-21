import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as font
import matplotlib


tw_font = font(fname="NotoSansTC-VariableFont_wght.ttf")

def addlabels(x,y, padding = 0.07):
  for i in range(len(x)):
    plt.text(i-padding, y[i]+5, y[i])

LABELS = ['負向','中立', '正向']
STAR_LABELS = ['一星', '二星', '三星', '四星', '五星']

def type1_origin_chart():
    df = pd.read_csv('new_data/docs_0804/Final_Origin/type1_comments_0804.csv')

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
    plt.savefig('new_data/docs_0804/Final_Origin/type1_origin_3.png')


def type1_gan_chart():
    df = pd.read_csv('new_data/docs_0819/Final_GPT4o_Mini/Type1_Result/gpt4o_type1_train_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('增生後資料(傳統露營)(GPT-4o mini)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_GPT4o_Mini/Type1_Result/type1_gpt4o_train.png')

def type2_origin_chart():
    df = pd.read_csv('new_data/docs_0724/Final_Origin/type2_comments.csv')

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
    plt.savefig('new_data/docs_0724/Final_Origin/type2_origin.png')


def type2_gan_chart():
    df = pd.read_csv('new_data/docs_0819/Final_GPT4o_Mini/Type2_Result/gpt4o_type2_train_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('增生後資料(懶人露營)(GPT-4o mini)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_GPT4o_Mini/Type2_Result/type2_gpt4o_train.png')


def type1_origin_split_chart(dataset):
    df = pd.read_csv(f"new_data/docs_0804/Final_Origin/Type1_Result/{dataset}.csv")
    values = []
    # values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] <= 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(傳統露營)(訓練集)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig(f'new_data/docs_0804/Final_Origin/Type1_Result/type1_{dataset}.png')


def type2_origin_split_chart(dataset):
    df = pd.read_csv(f"new_data/docs/Final_Origin/Type2_Result/{dataset}.csv")
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(懶人露營)(訓練集)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig(f'new_data/docs/Final_Origin/Type2_Result/type2_{dataset}.png')


def type1_origin_val_chart():
    df = pd.read_csv('new_data/docs_0819/Final_Origin/Type1_Result/type1_val_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(傳統露營)(驗證集)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/Type1_Result/type1_val_df.png')

def type1_origin_test_chart():
    df = pd.read_csv('new_data/docs_0819/Final_Origin/Type1_Result/type1_test_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(傳統露營)(測試集)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/Type1_Result/type1_test_df.png')

def type1_origin_train_chart():
    df = pd.read_csv('new_data/docs_0819/Final_Origin/type1_train_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(傳統露營)(訓練集)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/type1_train_df.png')


def type2_origin_val_chart():
    df = pd.read_csv('new_data/docs_0819/Final_Origin/Type2_Result/type2_val_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(懶人露營)(驗證集)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/Type2_Result/type2_val_df.png')

def type2_origin_test_chart():
    df = pd.read_csv('new_data/docs_0819/Final_Origin/Type2_Result/type2_test_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(懶人露營)(測試集)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/Type2_Result/type2_test_df.png')

def type2_origin_train_chart():
    df = pd.read_csv('new_data/docs_0819/Final_Origin/Type2_Result/type2_train_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料(懶人露營)(訓練集)', fontproperties=tw_font)
    plt.xlabel('評價類型', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/Type2_Result/type2_train_df.png')

if __name__ == "__main__":
    # type1_origin_chart()
    # type2_origin_chart()

    type1_gan_chart()
    type2_gan_chart()
    # type2_origin_val_chart()
    # type2_origin_test_chart()
    # type2_origin_train_chart()
    # type1_origin_split_chart("train_df_2")
    # type2_origin_split_chart("train_df")