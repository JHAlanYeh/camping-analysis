import requests
import json
import os
import jieba
from bs4 import BeautifulSoup

f = open('stopwords.txt', encoding="utf-8")
STOP_WORDS = []
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))

def asiayo_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "data")
    file = "camping_asiayo.json"
    total_type_1_count = 0
    total_type_2_count = 0

    file_path = os.path.join(dir_path, file)
    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    f.close()

    conform_data = filter(lambda x: x["disabled"] == 0 and ( x["type"] == 1 or  x["type"] == 2), data)
    conform_type_1_data = filter(lambda x: x["disabled"] == 0 and x["type"] == 1, data)
    conform_type_2_data = filter(lambda x: x["disabled"] == 0 and x["type"] == 2, data)
    conform_type_3_data = filter(lambda x: x["disabled"] == 0 and x["type"] == 3, data)
    conform_type_4_data = filter(lambda x: x["disabled"] == 0 and x["type"] == 4, data)
    
    uncategorized = filter(lambda x: x["type"] == 0 and x["disabled"] == 0, data)
    disabled = filter(lambda x: x["disabled"] == 1, data)

    print("Total Data:{}".format(len(data)))
    print("uncategorized data:{}".format(len(list(uncategorized))))
    print("conform data:{}".format(len(list(conform_data))))
    print("conform type 1 data:{}".format(len(list(conform_type_1_data))))
    print("conform type 2 data:{}".format(len(list(conform_type_2_data))))
    print("conform type 3 data:{}".format(len(list(conform_type_3_data))))
    print("conform type 4 data:{}".format(len(list(conform_type_4_data))))
    print("disabled data:{}".format(len(list(disabled))))

    for d in data:
        if d["disabled"] == 1 or d["type"] == 3 or d["type"] == 4:
            continue

        offset = 0
        code = d["code"]

        save_dir = os.path.join(dir_path, "asiayo_comments\\{house_type}".format(house_type=d["type"]))
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, 'comment_{code}.json'.format(code=code))

        comment_objs = []
        res = requests.get("https://web-api.asiayo.com/api/v1/bnbs/{code}?locale=zh-tw&currency=TWD&checkInDate=2024-02-03&checkOutDate=2024-02-04&people=1&adult=1&childAges=".format(code=code))
        res_json = res.json()
        if len(res_json["data"]) == 0:
            print("No data-{}".format(res_json["data"]["name"]))
        else:
            d["address"] = res_json["data"]["address"]["fullAddress"]
            d["description"] = res_json["data"]["description"].replace("\n", "").replace("\r", "").replace("\t", "")
            d["bnbType"] = res_json["data"]["bnbTypeName"]
            if "露營" not in d["bnbType"]:
                d["disabled"] = 1

        while True:
            res = requests.get("""https://web-api.asiayo.com/api/v1/bnbs/{code}/reviews?limit=10&offset={offset}&locale=zh-tw""".format(code=code, offset=offset))
            if res.status_code == 200:
                res_json = res.json()
                camping_comments = res_json["data"]["reviews"]
                if len(camping_comments) == 0:
                    break

                for comment in camping_comments:
                    content = comment["content"].replace("\n", "").replace("\r", "").replace("\t", "")
                    rating = comment["rating"]
                    publishedDate = comment["publishedDate"]

                    ws = jieba.lcut(content, cut_all=False)
                    new_ws = []
                    for word in ws:
                        if word not in STOP_WORDS:
                            new_ws.append(word)

                    comment_objs.append({
                        "content": content,
                        "rating": rating,
                        "publishedDate": publishedDate,
                        "tokenization": " | ".join(new_ws)
                    })
                offset += 10

        d["comments"] = comment_objs
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)
            print("save {file_name}".format(file_name=file_name))
        if d["type"] == 1:
            total_type_1_count += len(comment_objs)
        elif d["type"] == 2:
            total_type_2_count += len(comment_objs)
    print("Total Type 1 Count：{}".format(total_type_1_count))
    print("Total Type 2 Count：{}".format(total_type_2_count))


def easycamp_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "data")
    file = "camping_easycamp.json"
    total_type_1_count = 0
    total_type_2_count = 0
    comment_url = "https://www.easycamp.com.tw/store/purchase_rank/{code}/4/{page}"

    file_path = os.path.join(dir_path, file)
    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    f.close()

    conform_data = filter(lambda x: x["disabled"] == 0 and ( x["type"] == 1 or  x["type"] == 2), data)
    conform_type_1_data = filter(lambda x: x["disabled"] == 0 and x["type"] == 1, data)
    conform_type_2_data = filter(lambda x: x["disabled"] == 0 and x["type"] == 2, data)
    conform_type_3_data = filter(lambda x: x["disabled"] == 0 and x["type"] == 3, data)
    conform_type_4_data = filter(lambda x: x["disabled"] == 0 and x["type"] == 4, data)
    
    uncategorized = filter(lambda x: x["type"] == 0 and x["disabled"] == 0, data)
    disabled = filter(lambda x: x["disabled"] == 1, data)

    print("Total Data:{}".format(len(data)))
    print("uncategorized data:{}".format(len(list(uncategorized))))
    print("conform data:{}".format(len(list(conform_data))))
    print("conform type 1 data:{}".format(len(list(conform_type_1_data))))
    print("conform type 2 data:{}".format(len(list(conform_type_2_data))))
    print("conform type 3 data:{}".format(len(list(conform_type_3_data))))
    print("conform type 4 data:{}".format(len(list(conform_type_4_data))))
    print("disabled data:{}".format(len(list(disabled))))

    for d in data:
        code = d["code"]

        save_dir = os.path.join(dir_path, "easycamp_comments\\{house_type}".format(house_type=d["type"]))
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, 'comment_{code}.json'.format(code=code))

        comment_objs = []
        web = requests.get(d["url"])
        soup = BeautifulSoup(web.text, "html.parser")
        d["description"] = soup.select_one("#content_id").text.strip().replace("\n", "").replace("\r", "").replace("\t", "")
        d["address"] = soup.select_one(".camp-info .camp-add").text.strip()
        gps = soup.select_one(".camp-info div .camp-gps").text.strip()
        if gps != "":
            d["latitude"] = float(soup.select_one(".camp-info div .camp-gps").text.strip().split(',')[0])
            d["longitude"] = float(soup.select_one(".camp-info div .camp-gps").text.strip().split(',')[1])
        page = 1
        while True:
            camp_comment = requests.get(comment_url.format(code=code, page=page))
            soup = BeautifulSoup(camp_comment.text, "html.parser")
            comments = soup.select('#tab11 div .row')
            if len(comments) == 0:
                break
            for comment in comments:
                publishedDate = comment.select_one("div .evaluation-font-color div:nth-child(3)").text.replace("評價：", "").strip()
                contents = comment.select(".english-break-word")
                content = []
                if contents[0].text.strip() != "":
                    content.append(contents[0].text.strip().replace("\n", "").replace("\r", "").replace("\t", ""))
                if contents[1].text.strip() != "":
                    content.append(contents[1].text.strip().replace("\n", "").replace("\r", "").replace("\t", ""))

                content = "。".join(content)
                ws = jieba.lcut(content, cut_all=False)
                new_ws = []
                for word in ws:
                    if word not in STOP_WORDS:
                        new_ws.append(word)

                rating = len(comment.select(".fa.fa-star"))
                comment_objs.append({
                    "content": content,
                    "rating": rating,
                    "publishedDate": publishedDate,
                    "tokenization": " | ".join(new_ws)
                })
            page += 1

        d["comments"] = comment_objs
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)
            print("save {file_name}".format(file_name=file_name))

        if d["type"] == 1:
            total_type_1_count += len(comment_objs)
        elif d["type"] == 2:
            total_type_2_count += len(comment_objs)
    print("Total Type 1 Count：{}".format(total_type_1_count))
    print("Total Type 2 Count：{}".format(total_type_2_count))

if __name__ == "__main__":
    asiayo_comment_crawler()
    easycamp_comment_crawler()