import pandas as pd


df = pd.read_csv("new_data/docs/type1_comments_low.csv", encoding="utf-8-sig")
df = df[(df["content"].str.len() > 10) & (df["content"].str.len() < 512)]
df.to_csv('new_data/docs/type1_comments_low_v1.csv', index=False, encoding="utf-8-sig")

target_count = len(df[df["rating"] >= 4])
print(target_count)


df_mid = df[df["rating"] == 3]
df_low = df[df["rating"] <= 2]
print(f"原始：負向{len(df_low)}句，中立{len(df_mid)}句")

df_mid_seq = []
for index, row in df_mid.iterrows():
    df_mid_seq.append(row["sequence_num"])

df_low_seq = []
for index, row in df_low.iterrows():
    df_low_seq.append(row["sequence_num"])


# print(df_low_seq)
# print(df_mid_seq)

df_mid_gan_csv = pd.read_csv("new_data/docs/llama3_type1_mid_gan_df.csv", encoding="utf-8-sig")
df_mid_gan_csv[['sequence_num']] = df_mid_gan_csv[['sequence_num']].astype(int)
df_mid_gan_csv = df_mid_gan_csv[(df_mid_gan_csv["content"].str.len() > 10) & (df_mid_gan_csv["content"].str.len() < 512)]
print(len(df_mid_gan_csv))

df_low_gan_csv = pd.read_csv("new_data/docs/llama3_type1_low_gan_df.csv", encoding="utf-8-sig")
df_low_gan_csv[['sequence_num']] = df_low_gan_csv[['sequence_num']].astype(int)
df_low_gan_csv = df_low_gan_csv[(df_low_gan_csv["content"].str.len() > 10) & (df_low_gan_csv["content"].str.len() < 512)]
print(len(df_low_gan_csv))


df_mid_gan = []
for index, row in list(df_mid_gan_csv.iterrows()):
    if row["sequence_num"] in df_mid_seq:
        df_mid_gan.append(dict(row))

df_low_gan = []
for index, row in list(df_low_gan_csv.iterrows()):
    if row["sequence_num"] in df_low_seq:
        df_low_gan.append(dict(row))


print(len(df_mid_gan))
print(len(df_low_gan))

mid_gan_df = pd.json_normalize(df_mid_gan)
mid_gan_df.to_csv('new_data/docs/llama3_type1_mid_gan_df.csv', index=False, encoding="utf-8-sig")

low_gan_df = pd.json_normalize(df_low_gan)
low_gan_df.to_csv('new_data/docs/llama3_type1_low_gan_df.csv', index=False, encoding="utf-8-sig")