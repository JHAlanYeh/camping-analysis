import os
import json

dir_path = os.path.join(os.getcwd(), "new_data")
save_path = os.path.join(dir_path, "camping_region")

camping_reigon = []

for file in os.listdir(save_path):
    file_path = os.path.join(save_path, file)

    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)

    for d in data:
        name = d["name"].replace(":", "_").replace("\\", "_").replace("/", "_").replace("|", "_")
        if name + "\n" not in camping_reigon:
            camping_reigon.append(name + "\n")

camping_reigon = sorted(camping_reigon)

file_name = os.path.join(dir_path, 'all_region.txt')
with open(file_name, 'w', encoding="utf-8-sig") as f:
  f.writelines(camping_reigon)

