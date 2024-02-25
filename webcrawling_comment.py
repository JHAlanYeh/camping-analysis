import requests
import json
import os
import jieba
import math
from bs4 import BeautifulSoup
import random

jieba.load_userdict('custom_dict.txt')
# jieba.set_dictionary('dict.txt.big')


f = open('stopwords_zh_TW.dat.txt', encoding="utf-8")
STOP_WORDS = []
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))

f = open('stopwords.txt', encoding="utf-8")
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))

def asiayo_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "data")
    save_path = os.path.join(dir_path, "camping_region")
    file = "camping_asiayo.json"
    total_type_1_count = 0
    total_type_2_count = 0
    total_type_3_count = 0

    file_path = os.path.join(save_path, file)
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
    conform_type_3_count = len(list(conform_type_3_data))

    print("Total Data:{}".format(len(data)))
    print("uncategorized data:{}".format(len(list(uncategorized))))
    print("conform data:{}".format(len(list(conform_data))))
    print("conform type 1 data:{}".format(conform_type_1_count))
    print("conform type 2 data:{}".format(conform_type_2_count))
    print("conform type 3 data:{}".format(conform_type_3_count))
    print("conform type 4 data:{}".format(len(list(conform_type_4_data))))
    print("disabled data:{}".format(len(list(disabled))))

    for d in data:
        if d["disabled"] == 1 or d["type"] == 4:
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
            continue
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
        elif d["type"] == 3:
            total_type_3_count += len(comment_objs)
    print("Total Type 1 Count：{}".format(total_type_1_count))
    print("Total Type 2 Count：{}".format(total_type_2_count))
    print("Total Type 3 Count：{}".format(total_type_3_count))

    save_dir = os.path.join(dir_path, "comments_info")
    file_name = os.path.join(save_dir, 'asiayo_info.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        overview = {
            "conform_type_1": conform_type_1_count,
            "conform_type_2": conform_type_2_count,
            "conform_type_3": conform_type_3_count,
            "type_1_comments":total_type_1_count,
            "type_2_comments":total_type_2_count,
            "type_3_comments":total_type_3_count,
        }
        json.dump(overview, f, indent=4, ensure_ascii=False)


def asiayo_comment_tokenization():
    dir_path = os.path.join(os.getcwd(), "data")
    asiayo_dir = os.path.join(dir_path, "asiayo_comments")

    total_type_1_count = 0
    total_type_2_count = 0
    total_type_3_count = 0

    conform_type_1_count = 0
    conform_type_2_count = 0
    conform_type_3_count = 0

    for i in range(1, 4 + 1):
        asiayo_type_dir = os.path.join(asiayo_dir, str(i))
        if os.path.isdir(asiayo_type_dir):
            for file in os.listdir(asiayo_type_dir):
                file_path = os.path.join(asiayo_type_dir, file)
                f = open(file_path, encoding="utf-8-sig")
                data = json.load(f)
                for c in data["comments"]:
                    ws = jieba.lcut(c["content"], cut_all=False)
                    new_ws = []
                    for word in ws:
                        if word not in STOP_WORDS:
                            new_ws.append(word)
                    c["tokenization"] = " | ".join(new_ws)

                with open(file_path, 'w', encoding="utf-8-sig") as nf:
                    json.dump(data, nf, indent=4, ensure_ascii=False)
                    print("save {file_name}".format(file_name=file_path))

                if data["type"] == 1:
                    total_type_1_count += len(data["comments"])
                    conform_type_1_count += 1
                elif data["type"] == 2:
                    total_type_2_count += len(data["comments"])
                    conform_type_2_count += 1
                elif data["type"] == 3:
                    total_type_3_count += len(data["comments"])
                    conform_type_3_count += 1

    print("Total Type 1 Count：{}".format(total_type_1_count))
    print("Total Type 2 Count：{}".format(total_type_2_count))
    print("Total Type 3 Count：{}".format(total_type_3_count))

    save_dir = os.path.join(dir_path, "comments_info")
    file_name = os.path.join(save_dir, 'asiayo_info.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        overview = {
            "conform_type_1": conform_type_1_count,
            "conform_type_2": conform_type_2_count,
            "conform_type_3": conform_type_3_count,
            "type_1_comments":total_type_1_count,
            "type_2_comments":total_type_2_count,
            "type_3_comments":total_type_3_count,
        }
        json.dump(overview, f, indent=4, ensure_ascii=False)
                

def easycamp_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "data")
    save_path = os.path.join(dir_path, "camping_region")
    file = "camping_easycamp.json"
    total_type_1_count = 0
    total_type_2_count = 0
    total_type_3_count = 0
    comment_url = "https://www.easycamp.com.tw/store/purchase_rank/{code}/4/{page}"

    file_path = os.path.join(save_path, file)
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
    conform_type_3_count = len(list(conform_type_3_data))

    print("Total Data:{}".format(len(data)))
    print("uncategorized data:{}".format(len(list(uncategorized))))
    print("conform data:{}".format(len(list(conform_data))))
    print("conform type 1 data:{}".format(conform_type_1_count))
    print("conform type 2 data:{}".format(conform_type_2_count))
    print("conform type 3 data:{}".format(conform_type_3_count))
    print("conform type 4 data:{}".format(len(list(conform_type_4_data))))
    print("disabled data:{}".format(len(list(disabled))))

    for d in data:
        if d["disabled"] == 1 or d["type"] == 4:
            continue

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
        elif d["type"] == 3:
            total_type_3_count += len(comment_objs)
    print("Total Type 1 Count：{}".format(total_type_1_count))
    print("Total Type 2 Count：{}".format(total_type_2_count))
    print("Total Type 3 Count：{}".format(total_type_3_count))

    save_dir = os.path.join(dir_path, "comments_info")
    file_name = os.path.join(save_dir, 'easycamp_info.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        overview = {
            "conform_type_1": conform_type_1_count,
            "conform_type_2": conform_type_2_count,
            "conform_type_3": conform_type_3_count,
            "type_1_comments":total_type_1_count,
            "type_2_comments":total_type_2_count,
            "type_3_comments":total_type_3_count,
        }
        json.dump(overview, f, indent=4, ensure_ascii=False)

def easycamp_comment_tokenization():
    dir_path = os.path.join(os.getcwd(), "data")
    easycamp_dir = os.path.join(dir_path, "easycamp_comments")

    total_type_1_count = 0
    total_type_2_count = 0
    total_type_3_count = 0

    conform_type_1_count = 0
    conform_type_2_count = 0
    conform_type_3_count = 0

    for i in range(1, 4 + 1):
        easycamp_type_dir = os.path.join(easycamp_dir, str(i))
        if os.path.isdir(easycamp_type_dir):
            for file in os.listdir(easycamp_type_dir):
                file_path = os.path.join(easycamp_type_dir, file)
                f = open(file_path, encoding="utf-8-sig")
                data = json.load(f)
                for c in data["comments"]:
                    ws = jieba.lcut(c["content"], cut_all=False)
                    new_ws = []
                    for word in ws:
                        if word not in STOP_WORDS:
                            new_ws.append(word)
                    c["tokenization"] = " | ".join(new_ws)

                with open(file_path, 'w', encoding="utf-8-sig") as nf:
                    json.dump(data, nf, indent=4, ensure_ascii=False)
                    print("save {file_name}".format(file_name=file_path))

                if data["type"] == 1:
                    total_type_1_count += len(data["comments"])
                    conform_type_1_count += 1
                elif data["type"] == 2:
                    total_type_2_count += len(data["comments"])
                    conform_type_2_count += 1
                elif data["type"] == 3:
                    total_type_3_count += len(data["comments"])
                    conform_type_3_count += 1

    print("Total Type 1 Count：{}".format(total_type_1_count))
    print("Total Type 2 Count：{}".format(total_type_2_count))
    print("Total Type 3 Count：{}".format(total_type_3_count))

    save_dir = os.path.join(dir_path, "comments_info")
    file_name = os.path.join(save_dir, 'easycamp_info.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        overview = {
            "conform_type_1": conform_type_1_count,
            "conform_type_2": conform_type_2_count,
            "conform_type_3": conform_type_3_count,
            "type_1_comments":total_type_1_count,
            "type_2_comments":total_type_2_count,
            "type_3_comments":total_type_3_count,
        }
        json.dump(overview, f, indent=4, ensure_ascii=False)

def klook_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "data")
    save_path = os.path.join(dir_path, "camping_region")

    file = "camping_klook.json"
    total_type_1_count = 0
    total_type_2_count = 0
    total_type_3_count = 0

    file_path = os.path.join(save_path, file)
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
    conform_type_3_count = len(list(conform_type_3_data))

    print("Total Data:{}".format(len(data)))
    print("uncategorized data:{}".format(len(list(uncategorized))))
    print("conform data:{}".format(len(list(conform_data))))
    print("conform type 1 data:{}".format(conform_type_1_count))
    print("conform type 2 data:{}".format(conform_type_2_count))
    print("conform type 3 data:{}".format(conform_type_3_count))
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
    'Cookie': 'klk_currency=TWD; kepler_id=761ce78c-e8c4-4ed1-927d-fa3543f67eec; persisted_source=viviantrip.tw; k_tff_ch=aid_2620; _gcl_au=1.1.1239040881.1707752204; __lt__cid=e84c815c-4379-4377-9952-147530418dc4; __lt__cid.c83939be=e84c815c-4379-4377-9952-147530418dc4; _fwb=237XOdedFIxX1kiZGCqzRJx.1707752212523; _yjsu_yjad=1707752215.d311e5f3-cbae-44d7-8508-215e64247bbe; _tt_enable_cookie=1; _ttp=j6D-oFE6A5T30gS1d0a58mwBUT4; dable_uid=3221839.1700147487968; KOUNT_SESSION_ID=5CDFC80C89656DD1034CD99E124D4D63; clientside-cookie=737969e068a2f6407ca720bbfdc275feccaa31d9fef47c0a772d2e9e6048c024f2771ca16c3cbee8bbd936f8e66ee7b3e47a168fad8bd99079cb74cdbdcaf263883e67eec02f3fcd7e67a327d8edeaf3b9b3a0dadf3f9d66db4fc612f2a7a7e4c76c62b3ba8efdcc84e964afe34e405f3eb110689ef08761e04765cca78e1ae74b238a28fc71c058c6b783e6c7b4dc891228d5367b6db972032610; aid=2620; wid=2620; aid_query_string=aid%3D2620; affiliate_type=non-network; aid_extra=%7B%22aff_klick_id%22%3A%2259183430999-2620-0-4d4002f%22%2C%22affiliate_partner%22%3A%22%22%2C%22content%22%3A%22%22%7D; aid_campaign=aid=2620&utm_medium=affiliate-alwayson&utm_source=non-network&utm_campaign=2620; tr_update_tt=1707925831001; campaign_tag=klc_l1=Affiliate; traffic_retain=true; klk_ps=1; TNAHD=c42_1707926053329__c8109_1708008346057__c27456_1707926044903__c6488_1707925903585; klk_rdc=TW; __lt__sid=85b1645f-bf08c2ed; __lt__sid.c83939be=85b1645f-bf08c2ed; klk_ga_sn=2497024496..1708352802287; wcs_bt=s_2cb388a4aa34:1708352802; _gid=GA1.2.1833952077.1708352803; _dc_gtm_UA-86696233-1=1; _uetsid=e719e660cf3211eeb300b17c6029affa; _uetvid=d63970c0d13a11ed9dbf65988fc79cf5; _ga_FW3CMDM313=GS1.1.1708352802.10.0.1708352802.0.0.0; _ga_V8S4KC8ZXR=GS1.1.1708352803.10.0.1708352803.60.0.0; klk_i_sn=1340718126..1708352803824; _ga_TH9DNLM4ST=GS1.1.1708352806.10.1.1708352806.60.0.0; datadome=wYzO3xY7M2TOta3kawiUgulI0xma8o0sUxd6dHyKoPsHKpXaM5EGKcIpj0bv4EQR4fUWsXvQhnqhVveJZroYgFoOZceaZpqShIez7mLRzQbFPLGqNhdAIRLmK5TmUGc6; _ga=GA1.2.1057024504.1574164938; forterToken=1f3f45b14ff64e5b9e7ab1e8ff24de94_1708352804108__UDF4_20ck'
    }

    random_num = random.random(0, 10)
    if random_num % 2 == 0:
        data = sorted(data, key=lambda x: x["code"], reverse=True)

    for d in data:
        if d["disabled"] == 1 or d["type"] == 4:
            continue

        page = 1
        code = d["code"]
        print(code)

        save_dir = os.path.join(dir_path, "klook_comments\\{house_type}".format(house_type=d["type"]))
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, 'comment_{code}.json'.format(code=code))

        comment_objs = []

        while True:
            querystring = {"k_lang":"zh_TW","k_currency":"TWD","activity_id":code,"page":page,"limit":"8","sort_type":"0","only_image": "false"}
            res = requests.get(""" https://www.klook.com/v1/experiencesrv/activity/component_service/activity_reviews_list""", headers=headers, params=querystring)
            print(res.status_code)
            print(page)
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
            else:
                return

        d["comments"] = comment_objs
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)
            print("save {file_name}".format(file_name=file_name))
        
        if d["type"] == 1:
            total_type_1_count += len(comment_objs)
        elif d["type"] == 2:
            total_type_2_count += len(comment_objs)
        elif d["type"] == 3:
            total_type_3_count += len(comment_objs)
    print("Total Type 1 Count：{}".format(total_type_1_count))
    print("Total Type 2 Count：{}".format(total_type_2_count))
    print("Total Type 3 Count：{}".format(total_type_3_count))

    save_dir = os.path.join(dir_path, "comments_info")
    file_name = os.path.join(save_dir, 'klook_info.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        overview = {
            "conform_type_1": conform_type_1_count,
            "conform_type_2": conform_type_2_count,
            "conform_type_3": conform_type_3_count,
            "type_1_comments":total_type_1_count,
            "type_2_comments":total_type_2_count,
            "type_3_comments":total_type_3_count,
        }
        json.dump(overview, f, indent=4, ensure_ascii=False)


def klook_comment_tokenization():
    dir_path = os.path.join(os.getcwd(), "data")
    klook_dir = os.path.join(dir_path, "klook_comments")
    
    total_type_1_count = 0
    total_type_2_count = 0
    total_type_3_count = 0

    conform_type_1_count = 0
    conform_type_2_count = 0
    conform_type_3_count = 0

    for i in range(1, 4 + 1):
        klook_type_dir = os.path.join(klook_dir, str(i))
        if os.path.isdir(klook_type_dir):
            for file in os.listdir(klook_type_dir):
                file_path = os.path.join(klook_type_dir, file)
                f = open(file_path, encoding="utf-8-sig")
                data = json.load(f)
                for c in data["comments"]:
                    ws = jieba.lcut(c["content"], cut_all=False)
                    new_ws = []
                    for word in ws:
                        if word not in STOP_WORDS:
                            new_ws.append(word)
                    c["tokenization"] = " | ".join(new_ws)
                
                with open(file_path, 'w', encoding="utf-8-sig") as nf:
                    json.dump(data, nf, indent=4, ensure_ascii=False)
                    print("save {file_name}".format(file_name=file_path))

                if data["type"] == 1:
                    total_type_1_count += len(data["comments"])
                    conform_type_1_count += 1
                elif data["type"] == 2:
                    total_type_2_count += len(data["comments"])
                    conform_type_2_count += 1
                elif data["type"] == 3:
                    total_type_3_count += len(data["comments"])
                    conform_type_3_count += 1

    print("Total Type 1 Count：{}".format(total_type_1_count))
    print("Total Type 2 Count：{}".format(total_type_2_count))
    print("Total Type 3 Count：{}".format(total_type_3_count))

    save_dir = os.path.join(dir_path, "comments_info")
    file_name = os.path.join(save_dir, 'klook_info.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        overview = {
            "conform_type_1": conform_type_1_count,
            "conform_type_2": conform_type_2_count,
            "conform_type_3": conform_type_3_count,
            "type_1_comments":total_type_1_count,
            "type_2_comments":total_type_2_count,
            "type_3_comments":total_type_3_count,
        }
        json.dump(overview, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asiayo_comment_crawler()
    asiayo_comment_tokenization()
    easycamp_comment_crawler()
    easycamp_comment_tokenization()
    klook_comment_crawler()
    klook_comment_tokenization()