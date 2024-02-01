import requests
import json
import math
import os
from pathlib import Path

dir_path = os.path.join(os.getcwd(), "data")
json_list = os.listdir(dir_path)

for file in json_list:
    if ".json" not in file or "_finish" in file:
        continue
    file_path = os.path.join(dir_path, file)
    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    for d in data:
        offset = 0
        code = d["code"]

        save_dir = os.path.join(dir_path, "comments")
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(dir_path, '{save_dir}\\comment_{code}.json'.format(save_dir=save_dir, code=code))

        if os.path.exists(file_name):
            print("file existed")
            continue

        comment_objs = []
        res = requests.get("https://web-api.asiayo.com/api/v1/bnbs/{code}?locale=zh-tw&currency=TWD&checkInDate=2024-02-03&checkOutDate=2024-02-04&people=1&adult=1&childAges=".format(code=code))
        res_json = res.json()
        d["address"] = res_json["data"]["address"]["fullAddress"]

        while True:
            res = requests.get("""https://web-api.asiayo.com/api/v1/bnbs/{code}/reviews?limit=10&offset={offset}&locale=zh-tw""".format(code=code, offset=offset))
            if res.status_code == 200:
                res_json = res.json()
                camping_comments = res_json["data"]["reviews"]
                if len(camping_comments) == 0:
                    break

                for comment in camping_comments:
                    content = comment["content"]
                    rating = comment["rating"]
                    publishedDate = comment["publishedDate"]

                    comment_objs.append({
                        "content": content,
                        "rating": rating,
                        "publishedDate": publishedDate
                    })
                offset += 10


        d["comments"] = comment_objs
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)
            print("save {file_name}".format(file_name=file_name))
    

    f.close()
    print("{} has finished".format(file))
    p = Path(file_path)
    p.rename(Path(p.parent, f"{p.stem}_finish{p.suffix}"))