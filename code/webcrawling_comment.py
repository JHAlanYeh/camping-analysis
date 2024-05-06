import requests
import json
import os
import jieba
import math
from bs4 import BeautifulSoup
import random

# jieba.load_userdict('custom_dict.txt')
# jieba.set_dictionary('dict.txt.big')


# f = open('stopwords_zh_TW.dat.txt', encoding="utf-8")
# STOP_WORDS = []
# lines = f.readlines()
# for line in lines:
#     STOP_WORDS.append(line.rstrip('\n'))

# f = open('stopwords.txt', encoding="utf-8")
# lines = f.readlines()
# for line in lines:
#     STOP_WORDS.append(line.rstrip('\n'))

def asiayo_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "new_data")
    save_path = os.path.join(dir_path, "camping_region")
    file = "camping_asiayo.json"

    file_path = os.path.join(save_path, file)
    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    f.close()
    
    print(len(data))
    
    for d in data:
        offset = 0
        code = d["code"]

        save_dir = os.path.join(dir_path, "asiayo_comments")
        name = d["name"].replace(":", "_").replace("\\", "_").replace("/", "_").replace("|", "_")
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'{name}.json')

        comment_objs = []
        res = requests.get("https://web-api.asiayo.com/api/v1/bnbs/{code}?locale=zh-tw&currency=TWD&checkInDate=2024-02-03&checkOutDate=2024-02-04&people=1&adult=1&childAges=".format(code=code))
        res_json = res.json()
        if len(res_json["data"]) == 0:
            print("No data-{}".format(d["name"]))
            continue
        else:
            d["address"] = res_json["data"]["address"]["fullAddress"]
            d["description"] = res_json["data"]["description"].replace("\n", "").replace("\r", "").replace("\t", "")

        while True:
            res = requests.get("""https://web-api.asiayo.com/api/v1/bnbs/{code}/reviews?limit=10&offset={offset}&locale=zh-tw""".format(code=code, offset=offset))
            if res.status_code == 200:
                res_json = res.json()
                camping_comments = res_json["data"]["reviews"]
                if len(camping_comments) == 0:
                    break

                for comment in camping_comments:
                    content = comment["content"].replace("\n", "").replace("\r", "").replace("\t", "")
                    if len(content) <= 10 or len(content) >=512:
                        continue
                    rating = comment["rating"]
                    publishedDate = comment["publishedDate"]
                    comment_type = d["type"]

                    # ws = jieba.lcut(content, cut_all=False)
                    # new_ws = []
                    # for word in ws:
                    #     if word not in STOP_WORDS:
                    #         new_ws.append(word)

                    comment_objs.append({
                        "content": content,
                        "type": comment_type,
                        "rating": rating,
                        "publishedDate": publishedDate,
                        "tokenization": ""
                        # "tokenization": " | ".join(new_ws)
                    })
                offset += 10

        d["comments_count"] = len(comment_objs)
        d["comments"] = comment_objs
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)
            print("save {file_name}".format(file_name=file_name))


def asiayo_comment_tokenization():
    dir_path = os.path.join(os.getcwd(), "new_data")
    asiayo_dir = os.path.join(dir_path, "asiayo_comments")

    for i in range(1, 4 + 1):
        asiayo_type_dir = os.path.join(asiayo_dir, str(i))
        if os.path.isdir(asiayo_type_dir):
            for file in os.listdir(asiayo_type_dir):
                file_path = os.path.join(asiayo_type_dir, file)
                f = open(file_path, encoding="utf-8-sig")
                data = json.load(f)
                data["comments"] = list(filter(lambda x: len(x["content"]) > 10 and len(x["content"]) < 512, data["comments"]))
                # for c in data["comments"]:
                #     ws = jieba.lcut(c["content"], cut_all=False)
                #     new_ws = []
                #     for word in ws:
                #         if word not in STOP_WORDS:
                #             new_ws.append(word)
                #     c["tokenization"] = " | ".join(new_ws)

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


def easycamp_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "new_data")
    save_path = os.path.join(dir_path, "camping_region")
    file = "camping_easycamp.json"
    comment_url = "https://www.easycamp.com.tw/store/purchase_rank/{code}/4/{page}"

    file_path = os.path.join(save_path, file)
    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    f.close()

    print(len(data))
    
    for d in data:
        code = d["code"]
        name = d["name"].replace(":", "_").replace("\\", "_").replace("/", "_").replace("|", "_")

        save_dir = os.path.join(dir_path, "easycamp_comments")
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'{name}.json')

        comment_objs = []
        web = requests.get(d["url"])
        soup = BeautifulSoup(web.text, "html.parser")
        # if soup.select_one("#content_id") is not None:
        #     d["description"] = soup.select_one("#content_id").text.strip().replace("\n", "").replace("\r", "").replace("\t", "")
        # else:
        #     continue
        d["address"] = soup.select_one(".camp-info .camp-add").text.strip()
        gps_elem = soup.select_one(".camp-info div .camp-gps")
        if gps_elem is not None:
            gps = gps_elem.text.strip()
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

                comment_type = d["type"]
                content = "。".join(content)
                if len(content) <= 10 or len(content) >=512:
                        continue
                # ws = jieba.lcut(content, cut_all=False)
                # new_ws = []
                # for word in ws:
                #     if word not in STOP_WORDS:
                #         new_ws.append(word)

                rating = len(comment.select(".fa.fa-star"))
                comment_objs.append({
                    "content": content,
                    "type": comment_type,
                    "rating": rating,
                    "publishedDate": publishedDate,
                    "tokenization": ""
                    # "tokenization": " | ".join(new_ws)
                })
            page += 1

        d["comments_count"] = len(comment_objs)
        d["comments"] = comment_objs
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)
            print("save {file_name}".format(file_name=file_name))


def easycamp_comment_tokenization():
    dir_path = os.path.join(os.getcwd(), "new_data")
    easycamp_dir = os.path.join(dir_path, "easycamp_comments")

    for i in range(1, 4 + 1):
        easycamp_type_dir = os.path.join(easycamp_dir, str(i))
        if os.path.isdir(easycamp_type_dir):
            for file in os.listdir(easycamp_type_dir):
                file_path = os.path.join(easycamp_type_dir, file)
                f = open(file_path, encoding="utf-8-sig")
                data = json.load(f)
                data["comments"] = list(filter(lambda x: len(x["content"]) > 10 and len(x["content"]) < 512, data["comments"]))
                # for c in data["comments"]:
                #     ws = jieba.lcut(c["content"], cut_all=False)
                #     new_ws = []
                #     for word in ws:
                #         if word not in STOP_WORDS:
                #             new_ws.append(word)
                #     c["tokenization"] = " | ".join(new_ws)

                with open(file_path, 'w', encoding="utf-8-sig") as nf:
                    json.dump(data, nf, indent=4, ensure_ascii=False)
                    print("save {file_name}".format(file_name=file_path))


def klook_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "new_data")
    save_path = os.path.join(dir_path, "camping_region")

    file = "camping_klook.json"

    file_path = os.path.join(save_path, file)
    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    f.close()

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

    data = sorted(data, key=lambda x: x["code"], reverse=True)

    for d in data:
        page = 1
        code = d["code"]
        name = d["name"].replace(":", "_").replace("\\", "_").replace("/", "_").replace("|", "_")

        save_dir = os.path.join(dir_path, "klook_comments")
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'{name}.json')

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
                    if len(content) <= 10 or len(content) >=512:
                        continue
                    rating = math.floor(int(comment["rating"]) / 20)
                    publishedDate = comment["date"]

                    comment_type = d["type"]
                    # ws = jieba.lcut(content, cut_all=False)
                    # new_ws = []
                    # for word in ws:
                    #     if word not in STOP_WORDS:
                    #         new_ws.append(word)

                    comment_objs.append({
                        "content": content,
                        "type": comment_type,
                        "rating": rating,
                        "publishedDate": publishedDate,
                        "tokenization": ""
                        # "tokenization": " | ".join(new_ws)
                    })
                page += 1
            else:
                return

        d["comments_count"] = len(comment_objs)
        d["comments"] = comment_objs
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)
            print("save {file_name}".format(file_name=file_name))



def klook_comment_tokenization():
    dir_path = os.path.join(os.getcwd(), "new_data")
    klook_dir = os.path.join(dir_path, "klook_comments")

    for i in range(1, 4 + 1):
        klook_type_dir = os.path.join(klook_dir, str(i))
        if os.path.isdir(klook_type_dir):
            for file in os.listdir(klook_type_dir):
                file_path = os.path.join(klook_type_dir, file)
                f = open(file_path, encoding="utf-8-sig")
                data = json.load(f)
                data["comments"] = list(filter(lambda x: len(x["content"]) > 10 and len(x["content"]) <512, data["comments"]))
                # for c in data["comments"]:
                #     ws = jieba.lcut(c["content"], cut_all=False)
                #     new_ws = []
                #     for word in ws:
                #         if word not in STOP_WORDS:
                #             new_ws.append(word)
                #     c["tokenization"] = " | ".join(new_ws)
                
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



if __name__ == "__main__":
    # asiayo_comment_crawler()
    # asiayo_comment_tokenization()
    # easycamp_comment_crawler()
    # easycamp_comment_tokenization()
    klook_comment_crawler()
    # klook_comment_tokenization()