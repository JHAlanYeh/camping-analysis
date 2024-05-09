import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from bs4 import BeautifulSoup as Soup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from dateutil.relativedelta import relativedelta
import json
import jieba

# jieba.load_userdict('custom_dict.txt')
# # jieba.set_dictionary('dict.txt.big')

# f = open('stopwords_zh_TW.dat.txt', encoding="utf-8")
# STOP_WORDS = []
# lines = f.readlines()
# for line in lines:
#     STOP_WORDS.append(line.rstrip('\n'))

# f = open('stopwords.txt', encoding="utf-8")
# lines = f.readlines()
# for line in lines:
#     STOP_WORDS.append(line.rstrip('\n'))

browserOptions = webdriver.ChromeOptions()
browserOptions.add_argument("--start-maximized")

user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15"
browserOptions.add_argument('--user-agent={}'.format(user_agent))
# browserOptions.add_argument("--headless")
browserOptions.add_argument('--mute-audio')
# INFO_0 / WARNING_1 / ERROR_2 / FATAL_3 / DEFAULT_0
browserOptions.add_argument("log-level=3")
# ****************************************************************************** #
dir_path = os.path.join(os.getcwd(), "new_data")
save_path = os.path.join(dir_path, "camping_region")

flag = False
for file in os.listdir(dir_path):
    file_path = os.path.join(dir_path, file)
    print(file_path)
    if ".txt" not in file_path:
        continue
    # if "google" not in file_path:
    #     continue

    f = open(file_path, encoding="utf-8-sig")
    for d in f.readlines():
        
    # data = json.load(f)
    # data_reverse = sorted(data, key=lambda x: x["code"], reverse=True)
    # for d in data:
        # if d["disabled"] == 1 or d["type"] == 4:
        #     continue

        # if "google" not in file:
        #     continue

        # if "露營區" not in d["name"]:
        #     flag=True
        #     continue

        # if flag is False:
        #     continue

        # if "same_name" in d and "{}.json".format(d["same_name"]) in google_comment_files:
        #     continue

        # if "same_name" in d and d["same_name"] == "NA":
        #     continue

        camping_name = d.split(',')[0]
        camping_type = d.split(',')[1]
        print("========================")
        print(camping_name)

        save_dir = os.path.join(dir_path, "google_comments")
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, '{}.json'.format(camping_name))
        if os.path.isfile(file_name):
            continue

        browser = webdriver.Chrome(options=browserOptions)
        wait = WebDriverWait(browser, 20)
        default_url = "https://www.google.com/maps?authuser=0"
        browser.get(default_url)
        house_type = False

        
        # if "｜" in d["name"] and "asiayo" in file:
        #     camping_name = d["name"].split("｜")[0]
        # if "｜" in d["name"] and "klook" in file:
        #     camping_name = d["name"].split("｜")[1]
        # if " " in d["name"] and "easycamp" in file:
        #     camping_names = d["name"].replace("  ", " ").split(" ")
        #     camping_name = camping_names[len(camping_names)-1]

        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#searchboxinput")))
        browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys(camping_name)
        browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys(Keys.ENTER)

        time.sleep(5)
        try:
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div:nth-child(3) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db")))
            address = browser.find_element(By.CSS_SELECTOR, "div:nth-child(3) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db").text

            # wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div:nth-child(5) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db")))
            # phone = browser.find_element(By.CSS_SELECTOR, "div:nth-child(5) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db").text

            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".DUwDvf.lfPIob")))
            name = browser.find_element(By.CSS_SELECTOR, ".DUwDvf.lfPIob").text.replace(":", "_").replace("\\", "_").replace("/", "_").replace("|", "_")
            # d["same_name"] = name
            print(name)

            # with open(file_path, 'w', encoding="utf-8-sig") as f:
            #     json.dump(data, f, indent=4, ensure_ascii=False)
            #     print("save {file_name}".format(file_name=file_path))
        except Exception as e:
            print(e.args)
            continue
        
        # if "{}.json".format(name) in google_comment_files:
        #     continue

        google_map = browser.current_url

        absolute_position = (google_map.split("@")[1]).split(",")
        latitude = absolute_position[0]
        longitude = absolute_position[1]

        time.sleep(5)
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(2)")))
        title = browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(2)").text
        #QA0Szd > div > div > div.w6VYqd > div.bJzME.tTVLSc > div > div.e07Vkf.kA9KIf > div > div > div:nth-child(1) > div > div > button.hh2c6.G7m0Af
        if title != "評論":
            title = browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(3)").text
            if title != "評論":
                continue
            browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(3)").click()
            house_type = True
        else:
            browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(2)").click()
            house_type = False

        time.sleep(5)

        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.jANrlb > div.fontBodySmall")))
        reviews_count = browser.find_element(By.CSS_SELECTOR, "div.jANrlb > div.fontBodySmall").text.replace(" 篇評論", "").replace(",", "")
        print("總評論數：" + reviews_count)

        if int(reviews_count) > 5000:
            print("comments too much")
            continue

        current_reviews_count = 0

        pane = browser.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
        browser.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", pane)

        try:
            # 等待網頁元素的出現
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.m6QErb.Pf6ghf.KoSBEe.ecceSd.tLjsW > div.TrU0dc.kdfrQc > button")))
            # 找到排序方法的按鈕
            browser.find_element(By.CSS_SELECTOR, "div.m6QErb.Pf6ghf.KoSBEe.ecceSd.tLjsW > div.TrU0dc.kdfrQc > button").click()
        except Exception as e:
            # 等待網頁元素的出現
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf > div:nth-child(8) > button:nth-child(2)")))
            # 找到排序方法的按鈕
            browser.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf > div:nth-child(8) > button:nth-child(2)").click()


        time.sleep(5)
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#action-menu > div:nth-child(2)")))
        browser.find_element(By.CSS_SELECTOR, '#action-menu > div:nth-child(2)').click()
        
        while int(reviews_count) > current_reviews_count and current_reviews_count < 1800:
            pane = browser.find_element(By.CSS_SELECTOR, "div:nth-child(2) > div > div.e07Vkf.kA9KIf > div > div > div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
            browser.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", pane)

            time.sleep(5)

            # 獲取網頁原始碼
            soup = Soup(browser.page_source, "html.parser")

            # 獲取評論資料框架
            all_reviews = soup.select(".jftiEf.fontBodyMedium")
            # ar = all_reviews[0] # 第幾則評論
            current_reviews_count = len(all_reviews)
            print("目前爬到評論數：" + str(current_reviews_count))

        comment_objs = []

        time.sleep(5)

        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".jftiEf.fontBodyMedium")))
        all_reviews = browser.find_elements(By.CSS_SELECTOR, ".jftiEf.fontBodyMedium")

        for ar in all_reviews:
            try:
                if house_type == False:
                    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".rsqaWe")))
                    publishedDate = ar.find_element(By.CSS_SELECTOR, ".rsqaWe").text.replace("Google", "").replace("(","").replace(")","").strip()
                    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".kvMYJc")))
                    star = ar.find_element(By.CSS_SELECTOR, ".kvMYJc").get_attribute("aria-label").replace(" 顆星", "").strip()
                else:
                    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".xRkPPb")))
                    publishedDate = ar.find_element(By.CSS_SELECTOR, ".xRkPPb").text.replace("Google", "").replace("(","").replace(")","").strip()
                    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".fzvQIb")))
                    star = ar.find_element(By.CSS_SELECTOR, ".fzvQIb").text.replace("/5", "").strip()
            
                wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".MyEned")))
                content = ar.find_element(By.CSS_SELECTOR, ".MyEned").text
                if "全文" in content:
                    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".MyEned > span > button")))
                    ar.find_element(By.CSS_SELECTOR, ".MyEned > span > button").click()
                    content = ar.find_element(By.CSS_SELECTOR, ".MyEned > .wiI7pd").text

                content = content.replace("\n", "").replace("\r", "").replace("\t", "")

                # ws = jieba.lcut(content, cut_all=False)
                # new_ws = []
                # for word in ws:
                #     if word not in STOP_WORDS:
                #         new_ws.append(word)

                today = datetime.now()

                if "月" in publishedDate:
                    num = publishedDate.replace("個月前", "").strip()
                    createdDate = today - relativedelta(months=int(num))
                elif "天" in publishedDate:
                    num = publishedDate.replace("天前", "").strip()
                    createdDate = today - timedelta(days=int(num))
                elif "年" in publishedDate:
                    num = publishedDate.replace("年前", "").strip()
                    createdDate = today - relativedelta(years=int(num))
                elif "週" in publishedDate:
                    num = publishedDate.replace("週前", "").strip()
                    createdDate = today - timedelta(weeks=int(num))
                else:
                    createdDate = today

                comment_objs.append({
                    "content": content,
                    "rating": int(star),
                    "type": int(camping_type),
                    "publishedDate": createdDate.strftime("%Y/%m/%d"),
                    # "tokenization": " | ".join(new_ws)
                    "tokenization": ""
                })
            except Exception as e:
                print(e.args)
                continue
        
        sorted_comments = sorted(comment_objs, key=lambda d: d["publishedDate"], reverse=True) 
        camping = {
            "name": name,
            "address": address,
            "google_map": google_map,
            "latitude": latitude,
            "longitude": longitude,
            "type": 0,
            "count": len(sorted_comments),
            "comments" : sorted_comments,
            "phone": ""
        }

        
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            json.dump(camping, f, indent=4, ensure_ascii=False, sort_keys=False)
            print("save {file_name}".format(file_name=file_name))

        browser.quit()