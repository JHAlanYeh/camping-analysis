import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from dateutil.relativedelta import relativedelta
import json
import emoji
import shutil

# ****************************************************************************** #
root_path = os.path.join(os.getcwd(), "new_data")

target_directory = "summary_comments"
target_path = os.path.join(root_path, target_directory)

type1_comments = []
type2_comments = []
for file in os.listdir(target_path):
    if ".json" not in file:
        continue
    target_file_path = os.path.join(target_path, file)
    print(file)

    tf = open(target_file_path, encoding="utf-8-sig")
    target_data = json.load(tf)
    tf.close()

    if target_data[0]["type"] == 1:
        shutil.copyfile(target_file_path, os.path.join(target_path, "1//{}".format(file)))
    elif target_data[0]["type"] == 2:
        shutil.copyfile(target_file_path, os.path.join(target_path, "2//{}".format(file)))

    if len(target_data) < 100:
        continue
    for d in target_data:
        emojis = emoji.emoji_list(d["content"])
        for e in emojis:
            d["content"] = d["content"].replace(e["emoji"], "")

        if d["type"] == 1:
            type1_comments.append(d)
        elif d["type"] == 2:
            type2_comments.append(d)

df1 = pd.json_normalize(type1_comments)
df2 = pd.json_normalize(type2_comments)

# 1/2星 = -1(負面), 3星 = 0(中等), 4/5星 = 1(正面)
#define conditions
conditions1 = [
    df1['rating'] >= 4,
    df1['rating'] == 3,
    df1['rating'] <= 2,
]

conditions2 = [
df2['rating'] >= 4,
df2['rating'] == 3,
df2['rating'] <= 2,
]

#define results
results = [1, 0, -1]
df1['status'] = np.select(conditions1, results)
df1['origin'] = 1
df2['status'] = np.select(conditions2, results)
df2['origin'] = 1

print(df1.head(5))
print(df2.head(5))

df_path = os.path.join(root_path, "docs")

df1.to_csv(os.path.join(df_path, "type1_comments.csv"), encoding="utf-8-sig", index=False)
df2.to_csv(os.path.join(df_path, "type2_comments.csv"), encoding="utf-8-sig", index=False)
