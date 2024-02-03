import requests
import json
import math
import os
from pathlib import Path

dir_path = os.path.join(os.getcwd(), "data")
json_list = os.listdir(dir_path)
camping_objs = []

for file in json_list:
    if ".json" not in file:
        continue
    file_path = os.path.join(dir_path, file)
    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    for d in data:
       if next((item for item in camping_objs if item["code"] == d["code"]), None) is None:
          camping_objs.append(d)

    f.close()
    print("{} has finished".format(file))


sorted_camping_objs = sorted(camping_objs, key=lambda d: d["code"]) 
file_name = os.path.join(dir_path, 'camping_item.json')
with open(file_name, 'w', encoding="utf-8-sig") as f:
    json.dump(sorted_camping_objs, f, indent=4, ensure_ascii=False)
    print("save {file_name}".format(file_name=file_name))