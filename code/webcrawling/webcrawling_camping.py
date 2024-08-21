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
    url = "https://www.kkday.com/zh-tw/product/ajax_productlist"
    
    
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        'Accept': "application/json, text/plain, */*",
        'Cache-Control': "no-cache",
        'Accept-Language': "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6",
        'Accept-Encoding': "gzip, deflate, br, zstd",
        'Connection': "keep-alive",
        'cache-control': "no-cache",
        'Cookie': 'country_lang=zh-tw; currency=TWD; KKUD=538952ab1d2b5c12cede14260c12d8dc; _gcl_au=1.1.1513969168.1714914403; __lt__cid=17633441-f3ea-4475-ae99-9bf8f983b811; _fbp=fb.1.1714914403398.2105784403; _atrk_siteuid=NFyddQNFN-x4SQ52; rskxRunCookie=0; rCookie=6iqp0pzg7mn39v5dpb4rbhlvtjrcn5; CookieConsent={stamp:%270FEHhZQFbYiJi1L1mL6dpXipTO6DC5JUo/IyYNINn7dy7l4J/JM49Q==%27%2Cnecessary:true%2Cpreferences:true%2Cstatistics:true%2Cmarketing:true%2Cmethod:%27explicit%27%2Cver:1%2Cutc:1714914408154%2Cregion:%27tw%27}; CID=5193; UD1=DSA; UD2=prod; _gac_UA-49763723-1=1.1714914608.CjwKCAjw3NyxBhBmEiwAyofDYfULyA4qlKSCHenILVKqSy_TdmEjxWl6Oy-KA1epMczndHtkaD1UZxoCT00QAvD_BwE; _gac_UA-117438867-1=1.1714914608.CjwKCAjw3NyxBhBmEiwAyofDYfULyA4qlKSCHenILVKqSy_TdmEjxWl6Oy-KA1epMczndHtkaD1UZxoCT00QAvD_BwE; _hjSessionUser_628088=eyJpZCI6ImZiMmE1Y2NiLTI5MGItNTI4Yy1hOWVkLTM2MDA0Yjg2YjY0NSIsImNyZWF0ZWQiOjE3MTQ5MTQ0MDM0NzMsImV4aXN0aW5nIjp0cnVlfQ==; appier_utmz=%7B%22csr%22%3A%22(adwords%20gclid)%22%2C%22timestamp%22%3A1714914609%2C%22lcsr%22%3A%22google%22%7D; _gcl_aw=GCL.1714914622.CjwKCAjw3NyxBhBmEiwAyofDYfULyA4qlKSCHenILVKqSy_TdmEjxWl6Oy-KA1epMczndHtkaD1UZxoCT00QAvD_BwE; csrf_cookie_name=369e9272dd4ef3247d84855a78b71ab1; KKWEB=a%3A4%3A%7Bs%3A10%3A%22session_id%22%3Bs%3A32%3A%2260ce529663f2bc9b6dae3a460c5ee51d%22%3Bs%3A7%3A%22channel%22%3Bs%3A5%3A%22GUEST%22%3Bs%3A13%3A%22last_activity%22%3Bi%3A1715095892%3Bs%3A9%3A%22user_data%22%3Bs%3A0%3A%22%22%3B%7Daf01067317432d307ab774c54c0c1770; __lt__sid=a051619c-f20367a4; _hjSession_628088=eyJpZCI6ImU5ZGYxMzYyLWEyODQtNGIxZC05YTIxLTAzZDdlNmE2MWFiYyIsImMiOjE3MTUwOTU4OTQ0ODQsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=; _gid=GA1.2.263406581.1715095895; lastRskxRun=1715096148311; _ga=GA1.2.363046285.1714914403; _uetsid=e327de400c8611ef841e05dd3e2dad2c; _uetvid=51d425000ae011efb76c03e7a6092020; cto_bundle=cvlyTF91V3BjVU95bVBLdXBwVjVUa2FyN1NES3htNU5hMmdaYVBTUVJjMEJUcTVRQ29KNnZDMjd3TEpsMG9KNFBHSUxWODhEYmtjN3Nsd2UyJTJCQjlNQWNQN3lWTE5jazZNU3U0YXNxMERnJTJCJTJGU2pVdiUyQmZPZW1vVEolMkJQdE42V1FZWkJ5YlRPdSUyQmE4T1hHNlJkREd6SEhkUG9hMndVOUg5eHBWWFhFWEo0UnJNRzE1RGxPSGlCVVk5RWx3R0RDdmcyenNrOWhUUCUyRm1OaVhDSGltMWdWd0hyY0F5cjlWS3VmRkRaSzRkSWglMkJZcElMSlBXZzlxSGdmRE1ocWl5aDhlUnBkVjN2Q0dvVm9JeU01STNnOVZOSVVETWhhTHFqa1U4M21hcFBxNjYySDR4ald6ajVPNG9YSXBBJTJCREF6a2J5TmFrODZMbQ; datadome=l1LtOxHYZUdFB9KJ3vfA1~uFN47eJ3N4qMEjOxquuB_Jd4bzYpHQFBHwVSBdiHHxhWoUj9Ysg_pcELD003qi~TRxQGjRo9slts3l08lF4sFv_sjfhY_C0PT3GTnJtVoX; _ga_RJJY5WQFKP=GS1.1.1715095894.3.1.1715096194.13.0.0'
    }

    dir_path = os.path.join(os.getcwd(), "new_data//camping_region")
    file_name = os.path.join(dir_path, 'camping_kkday.json')
    f = open(file_name, encoding="utf-8-sig")
    camping_area_objs = json.load(f)
    new_obj = 0
    page = 1
    while True:
        querystring = {"country":"","city":"","keyword":"%E9%9C%B2%E7%87%9F","availstartdate":"","availenddate":"","cat":"","time": "", "glang":"", "sort": "popularity", "page":page, "row": 10,
                       "fprice": "*", "eprice": "*", "precurrency": "TWD", "csrf_token_name": "369e9272dd4ef3247d84855a78b71ab1"}
        res = requests.get(url.format(page), headers=headers)
        print(res.status_code)
        print(res)


        break
    print("==================== End KKday Crawler ====================")

if __name__ == "__main__":
    # asiayo_crawler()
    # easycamp_crawler()
    # klook_crawler()
    kkday_crawler()