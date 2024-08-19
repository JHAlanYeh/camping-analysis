import pandas as pd
import json
import os
import numpy as np
from sklearn.utils import shuffle

#### filter gan data ####
# df = pd.read_csv("new_data/docs_0819/Final_Taide/taide_type2_merge_df.csv")

# gan_df = df[df["origin"] == 0]

# low_df = gan_df[gan_df["rating"] <= 2]
# mid_df = gan_df[gan_df["rating"] == 3]
# print(len(mid_df), len(low_df))


# conditions = [
#     gan_df['rating'] >= 4,
#     gan_df['rating'] == 3,
#     gan_df['rating'] <= 2,
# ]

# results = ["正向", "中立", "負向"]
# gan_df['label'] = np.select(conditions, results)
# gan_df.to_csv(f"new_data/docs_0819/Final_Taide/taide_type2_gan_df.csv", index=False, encoding="utf-8-sig")



gan_df = pd.read_csv(f"new_data/docs_0819/Final_GPT35/gpt35_type1_gan_df.csv")
origin_train_df = pd.read_csv("new_data/docs_0819/Final_Origin/Type1_Result/type1_train_df.csv")

gan_mid_df = gan_df[gan_df["rating"] == 3]
gan_low_df = gan_df[gan_df["rating"] <= 2]

high_df = origin_train_df[origin_train_df["rating"] >= 4]
mid_df = origin_train_df[origin_train_df["rating"] == 3]
low_df = origin_train_df[origin_train_df["rating"] <= 2]
print(len(high_df), len(mid_df), len(low_df))



new_train_df = shuffle(pd.concat([high_df, mid_df, gan_mid_df.sample(len(high_df) - len(mid_df)), low_df, gan_low_df.sample(len(high_df) - len(low_df))]))
print(len(new_train_df))
new_train_df.to_csv("new_data/docs_0819/Final_GPT35/gpt35_type1_train_df.csv", index=False, encoding="utf-8-sig")