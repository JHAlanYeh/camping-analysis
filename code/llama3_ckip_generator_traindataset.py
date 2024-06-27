import pandas as pd
import numpy as np
import time


df = pd.read_csv("new_data/docs/type2_train_df.csv")
all_df = pd.read_csv("new_data/docs/llama3_type2_merge_df.csv")

target_count = len(df[df["label"] == 2])
print(f"需增生至{target_count}句")

df_mid = df[df["label"] == 1]
df_low = df[df["label"] == 0]
print(f"原始：負向{len(df_low)}句，中立{len(df_mid)}句")


mid_gan_df = pd.DataFrame()
low_gan_df = pd.DataFrame()


print("負向增生")
for index, row in df_low.iterrows():
    
    print("====================================")
    print(f"Origin: {row['sequence_num']} {row['content']}")

    print("\n增生文本如下：\n")
    print(f"序號：{row['sequence_num']}")

    filter_df = all_df[(all_df["sequence_num"] == row["sequence_num"]) & (all_df["origin"] == 0)]


    low_gan_df = pd.concat([low_gan_df, filter_df]) 
    

    low_gan_df.to_csv('new_data/docs/llama3_type2_low_gan_dataset.csv', index=False, encoding="utf-8-sig")
    print(f"目前增生數量： 增生{len(low_gan_df)}句，總共{len(low_gan_df) + len(df_low)}，目標{target_count}")


 
# # ========================================================= #
# 中立
for index, row in df_mid.iterrows():
    
    print("====================================")
    print(f"Origin: {row['sequence_num']} {row['content']}")

    print("\n增生文本如下：\n")
    print(f"序號：{row['sequence_num']}")

    filter_df = all_df[(all_df["sequence_num"] == row["sequence_num"]) & (all_df["origin"] == 0)]


    mid_gan_df = pd.concat([mid_gan_df, filter_df]) 
    

    mid_gan_df.to_csv('new_data/docs/llama3_type2_mid_gan_dataset.csv', index=False, encoding="utf-8-sig")
    print(f"目前增生數量： 增生{len(mid_gan_df)}句，總共{len(mid_gan_df) + len(df_mid)}，目標{target_count}")

