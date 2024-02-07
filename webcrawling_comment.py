import requests
import json
import os
from bs4 import BeautifulSoup

def asiayo_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "data")
    file = "camping_asiayo.json"
    total_comment_count = 0

    file_path = os.path.join(dir_path, file)
    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    f.close()
    for d in data:
        offset = 0
        code = d["code"]

        save_dir = os.path.join(dir_path, "asiayo_comments")
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(dir_path, '{save_dir}\\comment_{code}.json'.format(save_dir=save_dir, code=code))

        comment_objs = []
        res = requests.get("https://web-api.asiayo.com/api/v1/bnbs/{code}?locale=zh-tw&currency=TWD&checkInDate=2024-02-03&checkOutDate=2024-02-04&people=1&adult=1&childAges=".format(code=code))
        res_json = res.json()
        d["address"] = res_json["data"]["address"]["fullAddress"]
        d["description"] = res_json["data"]["description"]

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

        total_comment_count += len(comment_objs)
        print("Total Comment Count：{}".format(total_comment_count))


def easycamp_comment_crawler():
    dir_path = os.path.join(os.getcwd(), "data")
    file = "camping_easycamp.json"
    total_comment_count = 0
    comment_url = "https://www.easycamp.com.tw/store/purchase_rank/{code}/4/{page}"

    file_path = os.path.join(dir_path, file)
    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    f.close()

    for d in data:
        code = d["code"]
        save_dir = os.path.join(dir_path, "easycamp_comments")
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(dir_path, '{save_dir}\\comment_{code}.json'.format(save_dir=save_dir, code=code))
        comment_objs = []
        web = requests.get(d["url"])
        soup = BeautifulSoup(web.text, "html.parser")
        d["description"] = soup.select_one("#content_id").text.strip()
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
                    content.append(contents[0].text.strip())
                if contents[1].text.strip() != "":
                    content.append(contents[1].text.strip())

                rating = len(comment.select(".fa.fa-star"))
                comment_objs.append({
                    "content": "。".join(content),
                    "rating": rating,
                    "publishedDate": publishedDate
                })
            page += 1

        d["comments"] = comment_objs
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)
            print("save {file_name}".format(file_name=file_name))

        total_comment_count += len(comment_objs)
        print("Total Comment Count：{}".format(total_comment_count))

if __name__ == "__main__":
    asiayo_comment_crawler()
    easycamp_comment_crawler()