import requests
import json
import math
import os

dir_path = os.path.join(os.getcwd(), "data")
camping_area_url = "https://asiayo.com/zh-tw/view/tw/{city_code}/{code}/?tags=camping&isFromPropertyPage=true/"
offset = 0

while True:
    res = requests.get("""https://web-api.asiayo.com/api/v1/bnbs/search?locale=zh-tw&currency=TWD&checkInDate=2024-02-03&checkOutDate=2024-02-04&adult=1&quantity=1&type=country&country=tw&tags=camping&offset={offset}""".format(offset=offset))
    if res.status_code == 200:
        res_json = res.json()
        camping_areas = res_json["data"]["rows"]
        if len(camping_areas) == 0:
            break
        camping_area_objs = []
        for area in camping_areas:
            name = area["name"]
            city = area["address"]["city"]["name"]
            district = area["address"]["area"]["name"]
            code = area["id"]
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
                "url": url,
                "latitude":latitude,
                "longitude":longitude
            })
        
        file_name = os.path.join(dir_path, 'camping_{}.json'.format(math.floor(offset / 20)))
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(camping_area_objs, f, indent=4, ensure_ascii=False)
            print("save {file_name}".format(file_name=file_name))
        
        offset += 20
