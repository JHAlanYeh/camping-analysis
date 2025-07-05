import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datetime import datetime, timedelta

user_id = "U30bc9aaf24ea900745f69d036821e5e3"
channel_access_token = "kQ/vmM5sqBwC9Dyc4ODf1aD/CeTTHZKbU32UCZsxHKgsIglO7oqC29E6mgJcU8jpfT57f2Ordq0fglHqXXw33D5142fPiE6mCCT1m5PJ38Jnfu2qBhXp2m6q01jAZBI4LSc+qmSZzUze5AVUN4obEwdB04t89/1O/w1cDnyilFU="


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "llm_model\Llama-3-Taiwan-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=nf4_config
    # attn_implementation="flash_attention_2" # optional
)


file_name = "C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-management\\public\\data-sources\\comments.json"
comment_objs = []
with open(file_name, 'r', encoding="utf-8-sig") as file:
    comment_objs = json.load(file)


def get_week_date_range(year, week_number):
    # 確定該年的第一天是星期幾
    first_day_of_year = datetime(year, 1, 1)
    # 計算該年的第一週的起始日期
    first_week_start = first_day_of_year - timedelta(days=first_day_of_year.weekday())
    # 根據週數計算起始日期
    week_start = first_week_start + timedelta(weeks=week_number - 1)
    week_end = week_start + timedelta(days=6)
    return week_start.date(), week_end.date()

report_file_name = "C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-management\\public\\data-sources\\weekly_report.json"
report = []
with open(report_file_name, 'r', encoding="utf-8-sig") as file:
    report = json.load(file)


year = datetime.now().year
day_of_week = datetime.now().strftime("%A")

result = ""

report_flag = False

week_number = datetime.now().isocalendar()[1]
week_number = week_number - 1
print(year, week_number)

existed_data = list(filter(lambda x: x["week"] == str(year) + " W" + str(week_number), report))
if len(existed_data) > 0 and existed_data[0]["content"] != "-":
    print("W" + str(week_number) + "已產出報告")
    report_flag = True
else:
    result = ""
    start_date, end_date = get_week_date_range(year, week_number)
    print(f"Year {year}, Week {week_number}: {start_date} to {end_date}")

    str_start_date = datetime.strftime(start_date, '%Y/%m/%d')
    str_end_date = datetime.strftime(end_date, '%Y/%m/%d')

    data = list(filter(lambda x: x["publishedDate"] >= datetime.strftime(start_date, '%Y/%m/%d') and x["publishedDate"] <= datetime.strftime(end_date, '%Y/%m/%d'), comment_objs))

    print(f"有{len(data)}筆評論")

    if len(data) == 0:
        print(f"{year} W{week_number} 沒有評論")
    else:
        result = "\n\n".join(list(map(lambda x: x[1]["content"], enumerate(data))))
        # print(result)

        messages = [
            {
                "role": "role", 
                "content": """你是一個促進ESG永續發展的助手，我是一個露營地經營管理者，我會上傳當週的所有露營評論，請你根據以下的描述，將當週評論內可以改善的ESG相關問題列出來：
環境 (E - Environmental)：透過分析露營評論，找出影響露營場地環境的負面因素（如垃圾問題、生態破壞），幫助改善永續經營策略，
社會 (S - Social)：分析評論中的使用者體驗，提升服務品質、確保安全，讓露營成為更友善的社交活動，
治理 (G - Governance)：透過評論管理系統提供透明度和監管機制，讓我能即時處理負面評倫，促進良性經營。
如果您不知道，請說「很抱歉，我不知道」"""
            },
            {
                "role": "assistant", 
                "content": "好的，我了解你的需求了！請上傳當週的露營評論，我會根據 ESG 三大面向（環境、社會、治理）協助你整理出可以改善的相關問題，幫助你提升永續發展與經營品質。隨時可以上傳評論檔案，我準備好了！"
            },
            {
                "role": "user", 
                "content": result
            },
        ]

        input_ids  = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=8196,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        embeddings = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(embeddings, skip_special_tokens=True)
        print(response)

        if "很抱歉" in response:
            pass
        else:
            messages = [
                    {
                    "role": "role", 
                    "content": """你是一個促進ESG永續發展的助手，我是一個露營地經營管理者，我會上傳當週的所有露營評論，請你根據以下的描述，將當週評論內可以改善的ESG相關問題列出來：
    環境 (E - Environmental)：透過分析露營評論，找出影響露營場地環境的負面因素（如垃圾問題、生態破壞），幫助改善永續經營策略，
    社會 (S - Social)：分析評論中的使用者體驗，提升服務品質、確保安全，讓露營成為更友善的社交活動，
    治理 (G - Governance)：透過評論管理系統提供透明度和監管機制，讓我能即時處理負面評倫，促進良性經營。
    如果您不知道，請說「很抱歉，我不知道」"""
                },
                {
                    "role": "assistant", 
                    "content": "好的，我了解你的需求了！請上傳當週的露營評論，我會根據 ESG 三大面向（環境、社會、治理）協助你整理出可以改善的相關問題，幫助你提升永續發展與經營品質。隨時可以上傳評論檔案，我準備好了！"
                },
                {
                    "role": "user", 
                    "content": result
                },
            ]

            input_ids  = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                input_ids,
                max_new_tokens=8196,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            embeddings = outputs[0][input_ids.shape[-1]:]
            response = tokenizer.decode(embeddings, skip_special_tokens=True)

            print(response)

            if len(existed_data) > 0 and existed_data[0]["content"] == "-":
                if "很抱歉" not in response:
                    report.append({
                        "week": str(year) + " W" + str(week_number),
                        "start_date": str_start_date,
                        "end_date": str_end_date,
                        "content": response
                    })
            elif len(existed_data) == 0:
                if "很抱歉" not in response:
                    report.append({
                        "week": str(year) + " W" + str(week_number),
                        "start_date": str_start_date,
                        "end_date": str_end_date,
                        "content": response
                    })
            else:
                print("W" + str(week_number) + "已產出報告")
                report_flag = True

            report = sorted(report, key=lambda x: x["start_date"], reverse=True)

            with open('C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-demo\\data-sources\\weekly_report.json', 'w', encoding="utf-8-sig") as f:
                json.dump(report, f, indent=4, ensure_ascii=False, sort_keys=False)
                print("save file")

            with open('C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-management\\public\\data-sources\\weekly_report.json', 'w', encoding="utf-8-sig") as f:
                json.dump(report, f, indent=4, ensure_ascii=False, sort_keys=False)
                print("save file")



with open('C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-demo\\data-sources\\weekly_report.json', 'w', encoding="utf-8-sig") as f:
    json.dump(report, f, indent=4, ensure_ascii=False, sort_keys=False)
    print("save file")

with open('C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-management\\public\\data-sources\\weekly_report.json', 'w', encoding="utf-8-sig") as f:
    json.dump(report, f, indent=4, ensure_ascii=False, sort_keys=False)
    print("save file")


if report_flag == False:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + channel_access_token
    }
    push_text = f"""{str(year)} W{str(week_number)} 週報告已產出，請至後台查看。"""
    res = requests.post("https://api.line.me/v2/bot/message/push", 
                        json = {
                                "to": user_id, 
                                "messages": [
                                    {
                                        "type": "text", 
                                        "text": push_text
                                    }
                                ]
                        }, 
                        headers=headers)