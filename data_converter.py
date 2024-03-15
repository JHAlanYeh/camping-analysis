import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

directory = "clean_data"

def json2dataframe():
    data_1 = []
    data_2 = []

    if os.path.isdir(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            f = open(file_path, encoding="utf-8-sig")
            data = json.load(f)
            f.close()

            if data["type"] != 1 and data["type"] != 2:
                continue
            for c in data["comments"]:
                if c["type"] == 1:
                    data_1.append(c)
                elif c["type"] == 2:
                    data_2.append(c)

    df1 = pd.json_normalize(data_1)
    df2 = pd.json_normalize(data_2)

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
    df2['status'] = np.select(conditions2, results)

    print(df1.head(5))
    print(df2.head(5))

    df1.to_csv("docs//type1_{}.csv".format(datetime.now().strftime("%Y%m%d%H%M")), encoding="utf-8-sig")
    df2.to_csv("docs//type2_{}.csv".format(datetime.now().strftime("%Y%m%d%H%M")), encoding="utf-8-sig")


if __name__ == "__main__":
    json2dataframe()