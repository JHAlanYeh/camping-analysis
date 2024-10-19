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

browserOptions = webdriver.ChromeOptions()
browserOptions.add_argument("--start-maximized")

user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15"
browserOptions.add_argument('--user-agent={}'.format(user_agent))
# browserOptions.add_argument("--headless")
browserOptions.add_argument('--mute-audio')
# INFO_0 / WARNING_1 / ERROR_2 / FATAL_3 / DEFAULT_0
browserOptions.add_argument("log-level=3")
# ****************************************************************************** #

browser = webdriver.Chrome(options=browserOptions)
wait = WebDriverWait(browser, 20)
default_url = "https://www.google.com/maps?authuser=0"
browser.get(default_url)
house_type = False


wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#searchboxinput")))
browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys("蟬說：山中靜靜")
browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys(Keys.ENTER)

time.sleep(5)
try:
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div:nth-child(3) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db")))
    address = browser.find_element(By.CSS_SELECTOR, "div:nth-child(3) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db").text

    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".DUwDvf.lfPIob")))
    name = browser.find_element(By.CSS_SELECTOR, ".DUwDvf.lfPIob").text.replace(":", "_").replace("\\", "_").replace("/", "_").replace("|", "_")
    print(name)

 
except Exception as e:
    print(e.args)


google_map = browser.current_url

absolute_position = (google_map.split("@")[1]).split(",")
latitude = absolute_position[0]
longitude = absolute_position[1]

file_name = "C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-management\\src\\data\\comments.json"
comment_objs = []
with open(file_name, 'r', encoding="utf-8-sig") as file:
    comment_objs = json.load(file)


time.sleep(5)
wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(2)")))
title = browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(2)").text
if title != "評論":
    title = browser.find_element(By.CSS_SELECTOR, ".yx21af.XDi3Bc > div > button:nth-child(3)").text
    if title != "評論":
        print("no comments")
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

while int(reviews_count) > current_reviews_count:
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
            author = ar.find_element(By.CSS_SELECTOR, ".d4r55").text.strip()
            author_id = ar.find_element(By.CSS_SELECTOR, ".al6Kxe").get_attribute("data-href").split("/")[5].strip()
    
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".MyEned")))
        content = ar.find_element(By.CSS_SELECTOR, ".MyEned").text
        if "全文" in content:
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".MyEned > span > button")))
            ar.find_element(By.CSS_SELECTOR, ".MyEned > span > button").click()
            content = ar.find_element(By.CSS_SELECTOR, ".MyEned > .wiI7pd").text

        content = content.replace("\n", "").replace("\r", "").replace("\t", "")

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

        duplicate =  list(filter(lambda x: x['author_id'] == author_id, comment_objs))
        if len(duplicate) == 0:
            comment_objs.append({
                "author_id": author_id,
                "author": author,
                "content": content,
                "publishedDate": createdDate.strftime("%Y/%m/%d"),
                "rating": int(star),
            })
        else:
            print("duplicate")
            continue
    except Exception as e:
        print(e.args)
        print({
                "author_id": author_id,
                "author": author,
                "content": "",
                "publishedDate": createdDate.strftime("%Y/%m/%d"),
                "rating": int(star),
            })
        continue


sorted_comments = sorted(comment_objs, key=lambda d: d["publishedDate"], reverse=True) 

with open(file_name, 'w', encoding="utf-8-sig") as f:
    json.dump(sorted_comments, f, indent=4, ensure_ascii=False, sort_keys=False)
    print("save {file_name}".format(file_name=file_name))

browser.quit()