import requests
import json
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC


def asiayo_crawler():
    print("==================== Start Asiayo Crawler ====================")
    dir_path = os.path.join(os.getcwd(), "new_data//camping_region")
    camping_area_url = "https://asiayo.com/zh-tw/view/tw/{city_code}/{code}/?tags=camping&isFromPropertyPage=true/"
    offset = 0

    file_name = os.path.join(dir_path, 'camping_asiayo.json')
    f = open(file_name, encoding="utf-8-sig")
    camping_area_objs = json.load(f)
    new_obj = 0

    while True:
        res = requests.get("""https://web-api.asiayo.com/api/v1/bnbs/search?locale=zh-tw&currency=TWD&checkInDate=2024-08-16&checkOutDate=2024-08-17&adult=4&quantity=1&type=country&country=tw&tags=camping&offset={offset}""".format(offset=offset))
        if res.status_code == 200:
            res_json = res.json()
            camping_areas = res_json["data"]["rows"]
            if len(camping_areas) == 0:
                break
            
            for area in camping_areas:
                code = area["id"]
                bnb_type = area["typeName"]
                duplicated_item = next((item for item in camping_area_objs if item["code"] == code), None)
                if duplicated_item is not None:
                    if "bnbType" not in duplicated_item:
                        idx = camping_area_objs.index(duplicated_item)
                        camping_area_objs.pop(idx)
                        duplicated_item["bnbType"] = bnb_type
                        if "露營" not in duplicated_item["bnbType"]:
                            duplicated_item["disabled"] = 1
                        camping_area_objs.append(duplicated_item)
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
                disabled = 0
                if "露營" not in bnb_type:
                    disabled = 1

                camping_area_objs.append({
                    "code": code,
                    "name": name,
                    "city": city,
                    "district": district,
                    "region":city + district,
                    "url": url,
                    "latitude":latitude,
                    "longitude":longitude,
                    "type": 0,
                    "disabled": disabled,
                    "bnbType":bnb_type,
                })
                print("New Object:{}".format(name))
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

    dir_path = os.path.join(os.getcwd(), "new_data//camping_region")
    base_url = "https://www.easycamp.com.tw{relative_link}"

    file_name = os.path.join(dir_path, 'camping_easycamp.json')
    f = open(file_name, encoding="utf-8-sig")
    camping_area_objs = json.load(f)
    new_obj = 0

    for i in range(1,5):
        page = 1
        while True:
            web = requests.get("https://www.easycamp.com.tw/store/store_list/{i}/0/0/0/%5B%22default%22%5D/0/0/0/{page}".format(i=i, page=page))
            soup = BeautifulSoup(web.text, "html.parser")
            articles = soup.select('table > tbody > tr > td')
            if len(articles) == 0:
                break

            for article in articles:
                code = article.select_one('a').get('href').replace("/Store_", "").replace(".html", "")
                if next((item for item in camping_area_objs if item["code"] == code), None) is not None:
                        continue
                url = base_url.format(relative_link=article.select_one('a').get('href'))
                name = article.select_one('h2').text
                region = article.select_one('div.hvr-sweep-add > a').text
                camping_area_objs.append({
                    "code": code,
                    "name": name,
                    "region":region,
                    "url": url,
                    "type": 0,
                    "disabled": 0
                })

                print("New:" + name)
                new_obj += 1
            page = page + 1

    print("New:{}".format(str(new_obj)))
    print("Total:{}".format(len(camping_area_objs)))
    sorted_camping_objs = sorted(camping_area_objs, key=lambda d: d["code"]) 
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        json.dump(sorted_camping_objs, f, indent=4, ensure_ascii=False)
        print("save {file_name}".format(file_name=file_name))

    print("==================== End EasyCamp Crawler ====================")

def klook_crawler():
    print("==================== Start Klook Crawler ====================")
    dir_path = os.path.join(os.getcwd(), "new_data//camping_region")
    page = 1

    file_name = os.path.join(dir_path, 'camping_klook.json')
    f = open(file_name, encoding="utf-8-sig")
    camping_area_objs = json.load(f)
    new_obj = 0
    headers = {
    'User-Agent': "PostmanRuntime/7.20.1",
    'Accept': "*/*",
    'Cache-Control': "no-cache",
    'Postman-Token': "21e0f7f0-d1d5-4b5b-9bcc-1e3e9963ac72,170bb722-f70e-41a8-945c-37ded76fbfa7",
    'Host': "www.klook.com",
    'Accept-Language': "zh_TW",
    'Accept-Encoding': "gzip, deflate",
    'Connection': "keep-alive",
    'cache-control': "no-cache"
    }

    while True:
        querystring = {"page_size":"15","sort":"most_relevant","page_num":page,"country_ids":"14","special_option_ids":"","query":"%E9%9C%B2%E7%87%9F%E5%8D%80"}
        res = requests.get("""https://www.klook.com/v1/cardinfocenterservicesrv/search/platform/complete_search""", headers=headers, params=querystring)
        if res.status_code == 200:
            res_json = res.json()
            camping_areas = res_json["result"]["search_result"]["cards"]
            if len(camping_areas) == 0:
                break
            
            for area in camping_areas:
                code = area["data"]["vertical_id"]
                duplicated_item = next((item for item in camping_area_objs if item["code"] == code), None)
                if duplicated_item is not None:
                    continue
                name = area["data"]["title"]
                # city = area["data"]["city_name"]
                # # 經度
                # latitude = area["location"]["lat"]
                # # 緯度
                # longitude = area["location"]["lng"]
                url = area["data"]["deep_link"]
                disabled = 0

                camping_area_objs.append({
                    "code": code,
                    "name": name,
                    # "city": city,
                    # "district": district,
                    # "region":city + district,
                    "url": url,
                    "latitude":"",
                    "longitude":"",
                    "address": "",
                    "type": 0,
                    "disabled": disabled,
                })
                print("New Object:{}".format(name))
                new_obj += 1
        page += 1

    print("New:{}".format(str(new_obj)))
    print("Total:{}".format(len(camping_area_objs)))
    sorted_camping_objs = sorted(camping_area_objs, key=lambda d: d["code"]) 
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        json.dump(sorted_camping_objs, f, indent=4, ensure_ascii=False)
        print("save {file_name}".format(file_name=file_name))

    print("==================== End Klook Crawler ====================")


def kkday_crawler():
    print("==================== Start KKday Crawler ====================")
    url = "https://www.kkday.com/zh-tw/product/productlist?page={}&city=A01-001-00002%2CA01-001-00010%2CA01-001-00009%2CA01-001-00012%2CA01-001-00006%2CA01-001-00001%2CA01-001-00008%2CA01-001-00014%2CA01-001-00017%2CA01-001-00018%2CA01-001-00003%2CA01-001-00026%2CA01-001-00005%2CA01-001-00013%2CA01-001-00004%2CA01-001-00007%2CA01-001-00011%2CA01-001-00015%2CA01-001-00016&keyword=%E9%9C%B2%E7%87%9F&sort=prec"
    
    
    browserOptions = webdriver.ChromeOptions()
    browserOptions.add_argument("--start-maximized")

    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15"
    browserOptions.add_argument('--user-agent={}'.format(user_agent))
    # browserOptions.add_argument("--headless")
    browserOptions.add_argument('--mute-audio')
    # INFO_0 / WARNING_1 / ERROR_2 / FATAL_3 / DEFAULT_0
    browserOptions.add_argument("log-level=3")
    # ****************************************************************************** #

    dir_path = os.path.join(os.getcwd(), "new_data//camping_region")
    file_name = os.path.join(dir_path, 'camping_kkday.json')
    f = open(file_name, encoding="utf-8-sig")
    camping_area_objs = json.load(f)
    new_obj = 0
    page = 1
    while True:
        browser = webdriver.Chrome(options=browserOptions)
        wait = WebDriverWait(browser, 20)
        browser.get(url.format(page))
        
        print(browser.title)
        print(browser.current_url)
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".product-listview > a > .row")))
        campings = browser.find_element(By.CSS_SELECTOR, ".product-listview > a > .row")
        print(len(campings))
        break
    print("==================== End KKday Crawler ====================")

if __name__ == "__main__":
    # asiayo_crawler()
    # easycamp_crawler()
    klook_crawler()
    # kkday_crawler()