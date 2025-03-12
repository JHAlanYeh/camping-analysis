import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as font
import matplotlib
import numpy as np


tw_font = font(fname="NotoSansTC-VariableFont_wght.ttf")

def addlabels(x,y, padding = 0.07):
  for i in range(len(x)):
    plt.text(i-padding, y[i]+5, y[i])

LABELS = ['負向','中立', '正向']
STAR_LABELS = ['一星', '二星', '三星', '四星', '五星']

def type1_origin_chart():
    plt.close('all')
    df = pd.read_csv('new_data/docs_0819/Final_Origin/type1_comments_origin.csv')

    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar', color=['#4471C4'])

    plt.title('原始資料情緒分布圖(傳統露營)', fontproperties=tw_font)
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/type1_origin_3.png')


def type1_gan_chart():
    df = pd.read_csv('new_data/docs_0819/Final_TaiwanLLM/Type1_Result/taiwanllm_type1_train_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('增生後資料(傳統露營)(訓練集)', fontproperties=tw_font)
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/type1_gan_train.png')

def type2_origin_chart():
    plt.close('all')
    df = pd.read_csv('new_data/docs_0819/Final_Origin/type2_comments_origin.csv')

    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar', color=['#4471C4'])

    plt.title('原始資料情緒分布圖(懶人露營)', fontproperties=tw_font)
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/type2_origin_3.png')


def type2_gan_chart():
    df = pd.read_csv('new_data/docs_0819/Final_TaiwanLLM/Type2_Result/taiwanllm_type2_train_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('增生後資料(懶人露營)(訓練集)', fontproperties=tw_font)
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/type2_gan_train.png')


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
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
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
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig(f'new_data/docs/Final_Origin/Type2_Result/type2_{dataset}.png')



def type1_origin_dataset_chart(dataset):
    df = pd.read_csv('new_data/docs_0819/Final_Origin/Type1_Result/type1_val_df.csv')
    values = []
    values.append(len(df[df["status"] == -1]))
    values.append(len(df[df["status"] == 0]))
    values.append(len(df[df["status"] == 1]))

    ser = pd.Series(values, index=LABELS)
    type1_merge_df = pd.DataFrame(data=ser, index=LABELS)

    axes = type1_merge_df.plot(kind='bar')

    plt.title('原始資料切分資料集分布圖(傳統露營)', fontproperties=tw_font)
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/Type1_Result/type1_val_df.png')


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
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
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
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
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
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
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
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
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
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
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
    plt.xlabel('評論情緒', fontproperties=tw_font)
    plt.xticks(rotation=0)
    plt.ylabel('評論數量', fontproperties=tw_font)
    plt.legend('',frameon=False)
    addlabels(LABELS, values)
    for label in axes.get_xticklabels():
        label.set_fontproperties(tw_font)
    plt.savefig('new_data/docs_0819/Final_Origin/Type2_Result/type2_train_df.png')


def type1_origin_split():
    plt.close('all')
    datasets = ['訓練集', '測試集', '驗證集']

    # 每個類別 (負向、中立、正向) 在不同數據集的數量
    train_counts = [287, 243, 5098]  # 訓練集數據
    test_counts = [35, 35, 634]      # 測試集數據
    val_counts = [30, 47, 626]       # 驗證集數據

    # 組合數據以便堆疊
    data_counts = np.array([train_counts, test_counts, val_counts]).T  # 轉置成 (3,3) 矩陣

    # 類別標籤
    colors = ['#ffb549', '#ff585d', '#4471C4']  # 為不同情緒設置顏色


    # 用來記錄底部位置
    bottom_values = np.zeros(len(datasets))

    # 繪製堆疊長條圖
    for i, (label, color) in enumerate(zip(LABELS, colors)):
        plt.bar(datasets, data_counts[i], bottom=bottom_values, label=label, color=color)
        bottom_values += data_counts[i]  # 更新底部基準值

    # 添加數字標籤
    # for i in range(len(datasets)):
    #     y_offset = 0
    #     for j in range(len(datasets)):
    #         plt.text(i, y_offset + data_counts[j, i] / 2, str(data_counts[j, i]), 
    #                 ha='center', va='center', fontsize=10, color='black', fontproperties=tw_font)
    #         y_offset += data_counts[j, i]
    
    plt.xticks(fontproperties=tw_font)

    # 設定標題與標籤
    plt.xlabel("資料集類型", fontproperties=tw_font)
    plt.ylabel("評論數量", fontproperties=tw_font)
    plt.title("原始資料切分資料集分布圖(傳統露營)", fontproperties=tw_font)
    plt.legend(prop=tw_font)

    # 顯示圖表
    plt.savefig('new_data/docs_0819/Final_Origin/type1_origin_3_split.png')


def type2_origin_split():
    plt.close('all')
    datasets = ['訓練集', '測試集', '驗證集']

    # 每個類別 (負向、中立、正向) 在不同數據集的數量
    train_counts = [308, 139, 4639]  # 訓練集數據
    test_counts = [41, 18, 577]      # 測試集數據
    val_counts = [35, 19, 582]       # 驗證集數據

    # 組合數據以便堆疊
    data_counts = np.array([train_counts, test_counts, val_counts]).T  # 轉置成 (3,3) 矩陣

    # 類別標籤
    colors = ['#ffb549', '#ff585d', '#4471C4']  # 為不同情緒設置顏色


    # 用來記錄底部位置
    bottom_values = np.zeros(len(datasets))

    # 繪製堆疊長條圖
    for i, (label, color) in enumerate(zip(LABELS, colors)):
        plt.bar(datasets, data_counts[i], bottom=bottom_values, label=label, color=color)
        bottom_values += data_counts[i]  # 更新底部基準值

    # 添加數字標籤
    # for i in range(len(datasets)):
    #     y_offset = 0
    #     for j in range(len(datasets)):
    #         plt.text(i, y_offset + data_counts[j, i] / 2, str(data_counts[j, i]), 
    #                 ha='center', va='center', fontsize=10, color='black', fontproperties=tw_font)
    #         y_offset += data_counts[j, i]
    
    plt.xticks(fontproperties=tw_font)

    # 設定標題與標籤
    plt.xlabel("資料集類型", fontproperties=tw_font)
    plt.ylabel("評論數量", fontproperties=tw_font)
    plt.title("原始資料切分資料集分布圖(懶人露營)", fontproperties=tw_font)
    plt.legend(prop=tw_font)

    # 顯示圖表
    plt.savefig('new_data/docs_0819/Final_Origin/type2_origin_3_split.png')

if __name__ == "__main__":
    type1_origin_chart()
    type2_origin_chart()

    type1_origin_split()
    type2_origin_split()

    # type1_gan_chart()
    # type2_gan_chart()
    # type2_origin_val_chart()
    # type2_origin_test_chart()
    # type2_origin_train_chart()
    # type1_origin_split_chart("train_df_2")
    # type2_origin_split_chart("train_df")