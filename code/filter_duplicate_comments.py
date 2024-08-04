import pandas as pd
import json
import os
import numpy as np

df = pd.read_csv("new_data/docs_0724/Final_Origin/type1_comments.csv")

result = df.to_json(orient="records")
data = json.loads(result)

new_data = []
content = []
for row in data:
    if row['content'] not in content:
        new_data.append(row)
        content.append(row['content'])
    else:
        print(row['content'])

new_df = pd.json_normalize(new_data)

new_df['sequence_num'] = new_df.reset_index().index
conditions = [
    new_df['rating'] >= 4,
    new_df['rating'] == 3,
    new_df['rating'] <= 2,
]

results = ["正向", "中立", "負向"]
new_df['origin_label'] = np.select(conditions, results)

root_path = os.path.join(os.getcwd(), "new_data")
df_path = os.path.join(root_path, "docs_0724")

new_df.to_csv(os.path.join(df_path, "type1_comments_0728.csv"), encoding="utf-8-sig", index=False)
