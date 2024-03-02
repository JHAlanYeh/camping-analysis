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

def check_unfinished():
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
            
            print("============================")

            if "same_name" not in d:
                print(file_path)
                print(d["name"])
                continue
            if d["same_name"]  not in google_comment_files:
                print(file_path)
                print(d["name"])
                continue

        f.close()   


def revise_google_map():
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
    google_comment_files = os.listdir(os.path.join(dir_path, "google_comments"))
    for file in google_comment_files:
        file_path = os.path.join(dir_path, "google_comments\\{}".format(file))
        try:        
            f = open(file_path, encoding="utf-8-sig")
            google_info = json.load(f)

            if "phone" in google_info:
                continue

            browser = webdriver.Chrome(options=browserOptions)
            wait = WebDriverWait(browser, 20)
            default_url = "https://www.google.com/maps?authuser=0"
            browser.get(default_url)

            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#searchboxinput")))
            browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys(google_info["name"])
            browser.find_element(By.CSS_SELECTOR, "#searchboxinput").send_keys(Keys.ENTER)

            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div:nth-child(3) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db")))
            address = browser.find_element(By.CSS_SELECTOR, "div:nth-child(3) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db").text

            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div:nth-child(5) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db")))
            phone = browser.find_element(By.CSS_SELECTOR, "div:nth-child(5) > button > div > div.rogA2c > div.Io6YTe.fontBodyMedium.kR99db").text
            
            time.sleep(5)

            google_info["google_map"] = browser.current_url

            temp_phone = int(phone.replace(" ", ""))

            google_info["phone"] = phone
            sorted_comments = sorted(google_info["comments"], key=lambda d: d["publishedDate"], reverse=True) 
            google_info["comments"] = sorted_comments

            with open(file_path, 'w', encoding="utf-8-sig") as f:
                json.dump(google_info, f, indent=4, ensure_ascii=False)
                # print("save {file_name}".format(file_name=file_path))
        except ValueError:
            print("no phone number")
            print(file_path)
        except Exception as e:
            print("no google map")
            print(file_path)


if __name__ == "__main__":
    check_unfinished()
    # revise_google_map()
