import requests
import json
import os
import jieba
import math
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

    conform_type_1_count = len(list(conform_type_1_data))
    conform_type_2_count = len(list(conform_type_2_data))

    print("Total Data:{}".format(len(data)))
    print("uncategorized data:{}".format(len(list(uncategorized))))
    print("conform data:{}".format(len(list(conform_data))))
    print("conform type 1 data:{}".format(conform_type_1_count))
    print("conform type 2 data:{}".format(conform_type_2_count))
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
            print("No data-{}".format(d["name"]))
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

    save_dir = os.path.join(dir_path, "asiayo_comments")
    file_name = os.path.join(save_dir, 'asiayo_info.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        overview = {
            "conform_type_1": len(list(conform_type_1_data)),
            "conform_type_2": len(list(conform_type_2_data)),
            "type_1_comments":total_type_1_count,
            "type_2_comments":total_type_2_count
        }
        json.dump(overview, f, indent=4, ensure_ascii=False)


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

    conform_type_1_count = len(list(conform_type_1_data))
    conform_type_2_count = len(list(conform_type_2_data))

    print("Total Data:{}".format(len(data)))
    print("uncategorized data:{}".format(len(list(uncategorized))))
    print("conform data:{}".format(len(list(conform_data))))
    print("conform type 1 data:{}".format(conform_type_1_count))
    print("conform type 2 data:{}".format(conform_type_2_count))
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

    save_dir = os.path.join(dir_path, "easycamp_comments")
    file_name = os.path.join(save_dir, 'easycamp_info.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        overview = {
            "conform_type_1": len(list(conform_type_1_data)),
            "conform_type_2": len(list(conform_type_2_data)),
            "type_1_comments":total_type_1_count,
            "type_2_comments":total_type_2_count
        }
        json.dump(overview, f, indent=4, ensure_ascii=False)

def klook_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "data")
    file = "camping_klook.json"
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

    conform_type_1_count = len(list(conform_type_1_data))
    conform_type_2_count = len(list(conform_type_2_data))

    print("Total Data:{}".format(len(data)))
    print("uncategorized data:{}".format(len(list(uncategorized))))
    print("conform data:{}".format(len(list(conform_data))))
    print("conform type 1 data:{}".format(conform_type_1_count))
    print("conform type 2 data:{}".format(conform_type_2_count))
    print("conform type 3 data:{}".format(len(list(conform_type_3_data))))
    print("conform type 4 data:{}".format(len(list(conform_type_4_data))))
    print("disabled data:{}".format(len(list(disabled))))

    headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    'Accept': "application/json, text/plain, */*",
    'Cache-Control': "no-cache",
    'Accept-Language': "zh_TW",
    'Accept-Encoding': "gzip, deflate",
    'Connection': "keep-alive",
    'cache-control': "no-cache",
    'Cookie': '_gid=GA1.2.2052794379.1707979075; klk_currency=TWD; kepler_id=ac63ff44-622e-47dc-aa4c-d3c0740a2d81; klk_rdc=TW; referring_domain_channel=seo; persisted_source=www.google.com; k_tff_ch=google_seo; _gcl_au=1.1.714077816.1708053514; _yjsu_yjad=1708053513.571571cc-3053-4588-95eb-fa88d2d58928; tr_update_tt=1708053514386; campaign_tag=klc_l1%3DSEO; dable_uid=64605085.1599094788315; _fwb=545IkjHhZxzAgfpUqwyoY2.1708053514425; traffic_retain=true; JSESSIONID=12FFEDD5E9AB383E10631BCFA1706A11; KOUNT_SESSION_ID=12FFEDD5E9AB383E10631BCFA1706A11; clientside-cookie=52476df9f3dca53892ea252f3743f94dae71ddfe801d6d6b61f1389f8cbd5c539316b338811f8ed63204cd1525fae4937d931db74043a02b78b7ae1d45de372f57caf66b29bdd3ec7a72e4240a46c5fe19467037b3672bcea12939d4c4a56ca3e2250d20c76cd1b946cdc861eca5563520ed561345454d33fd8828c12a770f1c890766c868a29b466e32fe24f669d1d2386847d83c94682037c1; klk_ga_sn=7675013114..1708069110168; wcs_bt=s_2cb388a4aa34:1708069110; _ga=GA1.1.1214317497.1633311520; _uetsid=1107bfc0cc7a11ee9e27fb8840a58085; _uetvid=1107e6d0cc7a11ee9ecc6bc3a37c7788; _ga_V8S4KC8ZXR=GS1.1.1708069111.2.0.1708069111.60.0.0; ftr_blst_1h=1708069111675; _ga_TH9DNLM4ST=GS1.1.1708069112.2.1.1708069112.60.0.0; forterToken=57f07b0b7182495598b8336ab3f7fa6f_1708069111420__UDF43-m4_20ck_; datadome=prkDBjFDk8gEA~zYcQGuk_4XQkTGPnKS~9H4nT2c4BY75kWrNZ5~G8HnYFXrdtmNrbfVV8pZKfAgpseNQcJzesiO43Ml4wWq7zrgIHC9KJYf2hfV_EVolQR5nUtzxv5o; _ga_FW3CMDM313=GS1.1.1708069111.2.0.1708069216.0.0.0; klk_i_sn=3194966446..1708069220250'
    }

    for d in data:
        if d["disabled"] == 1 or d["type"] == 3 or d["type"] == 4:
            continue

        page = 1
        code = d["code"]

        save_dir = os.path.join(dir_path, "klook_comments\\{house_type}".format(house_type=d["type"]))
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, 'comment_{code}.json'.format(code=code))

        comment_objs = []

        while True:
            querystring = {"k_lang":"zh_TW","k_currency":"TWD","activity_id":code,"page":page,"limit":"8","sort_type":"0","only_image": "false"}
            res = requests.get(""" https://www.klook.com/v1/experiencesrv/activity/component_service/activity_reviews_list""", headers=headers, params=querystring)
            if res.status_code == 200:
                res_json = res.json()
                camping_comments = res_json["result"]["item"]
                if camping_comments == None:
                    break

                for comment in camping_comments:
                    content = comment["content"].replace("\n", "").replace("\r", "").replace("\t", "")
                    rating = math.floor(int(comment["rating"]) / 20)
                    publishedDate = comment["date"]

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

    save_dir = os.path.join(dir_path, "klook_comments")
    file_name = os.path.join(save_dir, 'klook_info.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        overview = {
            "conform_type_1": len(list(conform_type_1_data)),
            "conform_type_2": len(list(conform_type_2_data)),
            "type_1_comments":total_type_1_count,
            "type_2_comments":total_type_2_count
        }
        json.dump(overview, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asiayo_comment_crawler()
    easycamp_comment_crawler()
    klook_comment_crawler()