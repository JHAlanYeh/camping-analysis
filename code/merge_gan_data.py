import pandas as pd

type1_mid_gan_df = pd.read_csv("new_data/docs/llama3_type1_mid_gan_dataset.csv", encoding="utf-8-sig")
type1_low_gan_df = pd.read_csv("new_data/docs/llama3_type1_low_gan_dataset.csv", encoding="utf-8-sig")
type1_origin_df = pd.read_csv("new_data/docs/type1_train_df.csv", encoding="utf-8-sig")

type1_positive = len(type1_origin_df[type1_origin_df["rating"] >= 4])
type1_negative = len(type1_origin_df[type1_origin_df["rating"] <= 2]) + len(type1_low_gan_df[type1_low_gan_df["rating"] <= 2])
type1_mid = len(type1_origin_df[type1_origin_df["rating"] == 3]) + len(type1_mid_gan_df[type1_mid_gan_df["rating"] == 3])
print(type1_positive, type1_negative, type1_mid)

if type1_negative > type1_positive:
    type1_low_gan_df = type1_low_gan_df.sample(type1_positive - len(type1_origin_df[type1_origin_df["rating"] <= 2]))

if type1_mid > type1_positive:
    type1_mid_gan_df = type1_mid_gan_df.sample(type1_positive - len(type1_origin_df[type1_origin_df["rating"] == 3]))

type1_positive = len(type1_origin_df[type1_origin_df["rating"] >= 4])
type1_negative = len(type1_origin_df[type1_origin_df["rating"] <= 2]) + len(type1_low_gan_df[type1_low_gan_df["rating"] <= 2])
type1_mid = len(type1_origin_df[type1_origin_df["rating"] == 3]) + len(type1_mid_gan_df[type1_mid_gan_df["rating"] == 3])
print(type1_positive, type1_negative, type1_mid)

type1_merge_llama3_df = pd.concat([type1_mid_gan_df, type1_low_gan_df, type1_origin_df])
type1_merge_llama3_df.to_csv('new_data/docs/Final_Llama3/llama3_type1_merge_train_dataset.csv', index=False, encoding="utf-8-sig")


# type2_mid_gan_df = pd.read_csv("new_data/docs/llama3_type2_mid_gan_dataset.csv", encoding="utf-8-sig")
# type2_low_gan_df = pd.read_csv("new_data/docs/llama3_type2_low_gan_dataset.csv", encoding="utf-8-sig")
# type2_origin_df = pd.read_csv("new_data/docs/type2_train_df.csv", encoding="utf-8-sig")

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
# type2_merge_llama3_df.to_csv('new_data/docs/Final_Llama3/llama3_type2_merge_train_dataset.csv', index=False, encoding="utf-8-sig")

