import os
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from dateutil.relativedelta import relativedelta
from selenium import webdriver
import time

browserOptions = webdriver.ChromeOptions()
browserOptions.add_argument("--start-maximized")

user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15"
browserOptions.add_argument('--user-agent={}'.format(user_agent))
# browserOptions.add_argument("--headless")
browserOptions.add_argument('--mute-audio')
# INFO_0 / WARNING_1 / ERROR_2 / FATAL_3 / DEFAULT_0
browserOptions.add_argument("log-level=3")
# ****************************************************************************** #

dir_path = os.path.join(os.getcwd(), "data")
save_path = os.path.join(dir_path, "camping_region")
google_comment_files = os.listdir(os.path.join(dir_path, "google_comments"))

for file in os.listdir(save_path):
    file_path = os.path.join(save_path, file)
    print(file_path)

    f = open(file_path, encoding="utf-8-sig")
    data = json.load(f)
    for d in data:
        if d["disabled"] == 1 or d["type"] == 4:
            continue

        browser = webdriver.Chrome(options=browserOptions)
        wait = WebDriverWait(browser, 20)
        default_url = "https://www.google.com/maps?authuser=0"
        browser.get(default_url)
        house_type = False
        
        camping_name = d["name"]
        
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#searchboxinput")))
        browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys(camping_name)
        browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys(Keys.ENTER)

        time.sleep(5)
        try:
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".Io6YTe.fontBodyMedium.kR99db")))
            address = browser.find_element(By.CSS_SELECTOR, ".Io6YTe.fontBodyMedium.kR99db").text

            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".DUwDvf.lfPIob")))
            name = browser.find_element(By.CSS_SELECTOR, ".DUwDvf.lfPIob").text.replace(":", "_").replace("\\", "_").replace("/", "_").replace("|", "_")
            d["same_name"] = name

        except Exception as e:
            if "same_name" in d and "{}.json".format(d["same_name"]) in google_comment_files:
                print(d["same_name"])
            continue
        