import requests
import json
import os
from bs4 import BeautifulSoup


def asiayo_crawler():
    print("==================== Start Asiayo Crawler ====================")
    dir_path = os.path.join(os.getcwd(), "data")
    camping_area_url = "https://asiayo.com/zh-tw/view/tw/{city_code}/{code}/?tags=camping&isFromPropertyPage=true/"
    offset = 0

    file_name = os.path.join(dir_path, 'camping_asiayo.json')
    f = open(file_name, encoding="utf-8-sig")
    camping_area_objs = json.load(f)
    new_obj = 0
    while True:
        res = requests.get("""https://web-api.asiayo.com/api/v1/bnbs/search?locale=zh-tw&currency=TWD&checkInDate=2024-03-21&checkOutDate=2024-03-22&adult=4&quantity=1&type=country&country=tw&tags=camping&offset={offset}""".format(offset=offset))
        if res.status_code == 200:
            res_json = res.json()
            camping_areas = res_json["data"]["rows"]
            if len(camping_areas) == 0:
                break
            
            for area in camping_areas:
                code = area["id"]
                if next((item for item in camping_area_objs if item["code"] == code), None) is not None:
                    continue
                name = area["name"]
                city = area["address"]["city"]["name"]
                district = area["address"]["area"]["name"]
                city_code = area["address"]["city"]["urlName"]
                # 經度
                latitude = area["location"]["lat"]
                # 緯度
                longitude = area["location"]["lng"]
                url = camping_area_url.format(city_code=city_code, code=code)

                camping_area_objs.append({
                    "code": code,
                    "name": name,
                    "city": city,
                    "district": district,
                    "region":city + district,
                    "url": url,
                    "latitude":latitude,
                    "longitude":longitude
                })
                new_obj += 1
        offset += 20
            

    print("New:{}".format(str(new_obj)))
    print("Total:{}".format(len(camping_area_objs)))
    sorted_camping_objs = sorted(camping_area_objs, key=lambda d: d["code"]) 
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        json.dump(sorted_camping_objs, f, indent=4, ensure_ascii=False)
        print("save {file_name}".format(file_name=file_name))

    print("==================== End Asiayo Crawler ====================")


def easycamp_crawler():
    print("==================== Start EasyCamp Crawler ====================")

    dir_path = os.path.join(os.getcwd(), "data")
    base_url = "https://www.easycamp.com.tw{relative_link}"

    file_name = os.path.join(dir_path, 'camping_easycamp.json')
    camping_area_objs = []

    main_url = "https://www.easycamp.com.tw/AC_Stores.html"
    web = requests.get(main_url)
    soup = BeautifulSoup(web.text, "html.parser")
    all_country = soup.select("[id^='store_menu_']")
    for country in all_country:
        link =  base_url.format(relative_link=country.select_one("li a").get('href'))
        camp_region = requests.get(link)
        soup = BeautifulSoup(camp_region.text, "html.parser")
        articles = soup.select('article')
        for article in articles:
            code = article.select_one('a').get('href').replace("/Store_", "").replace(".html", "")
            url = base_url.format(relative_link=article.select_one('a').get('href'))
            name = article.select_one('h3 a').text
            region = article.select_one('.companyinfo a').text
            camping_area_objs.append({
                "code": code,
                "name": name,
                "region":region,
                "url": url,
            })


    print("Total:{}".format(len(camping_area_objs)))
    sorted_camping_objs = sorted(camping_area_objs, key=lambda d: d["url"]) 
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        json.dump(sorted_camping_objs, f, indent=4, ensure_ascii=False)
        print("save {file_name}".format(file_name=file_name))

    print("==================== End EasyCamp Crawler ====================")


if __name__ == "__main__":
    # asiayo_crawler()
    easycamp_crawler()
