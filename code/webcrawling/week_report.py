import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datetime import datetime, timedelta


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

week_number = datetime.now().isocalendar()[1]
week_number = week_number - 1
print(year, week_number)

existed_data = list(filter(lambda x: x["week"] == str(year) + " W" + str(week_number), report))
if len(existed_data) > 0 and existed_data[0]["content"] != "-":
    print("W" + str(week_number) + "已產出報告")
else:
    result = ""
    start_date, end_date = get_week_date_range(year, week_number)
    print(f"Year {year}, Week {week_number}: {start_date} to {end_date}")

    str_start_date = datetime.strftime(start_date, '%Y/%m/%d')
    str_end_date = datetime.strftime(end_date, '%Y/%m/%d')

    data = list(filter(lambda x: x["publishedDate"] >= datetime.strftime(start_date, '%Y/%m/%d') and x["publishedDate"] <= datetime.strftime(end_date, '%Y/%m/%d'), comment_objs))

    if len(data) == 0 and len(existed_data) == 0:
        report.append({
            "week": str(year) + " W" + str(week_number),
            "start_date": str_start_date,
            "end_date": str_end_date,
            "content": "-"  
        })
    elif len(data) == 0 and len(existed_data) > 0:
        if existed_data[0]["content"] == "-":
            report.append({
                "week": str(year) + " W" + str(week_number),
                "start_date": str_start_date,
                "end_date": str_end_date,
                "content": "-"  
            })
    else:
        print("W" + str(week_number) + "無數據可產出")

    result = "\n".join(list(map(lambda x: str(x[0]+1) + "." + x[1]["content"], enumerate(data))))

    print(result)

    messages = [
        {
            "role": "role", 
            "content": """你是一個露營地經營管理者，我會上傳當週的所有露營評論，請你看完思考後提出露營地可以改進或優化的方向的總結，不需要多加贅字，僅需列出可優化的地方即可。
            如果您不知道，請說「很抱歉，我不知道」"""
        },
        {
            "role": "assistant", 
            "content": "好的，請您上傳露營的評論內容。"
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

    report = sorted(report, key=lambda x: x["start_date"], reverse=True)
    print(report)

    with open('C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-demo\\data-sources\\weekly_report.json', 'w', encoding="utf-8-sig") as f:
        json.dump(report, f, indent=4, ensure_ascii=False, sort_keys=False)
        print("save file")

    with open('C:\\Users\\Alan\\Documents\\Projects\\NCKU\\camping-management\\public\\data-sources\\weekly_report.json', 'w', encoding="utf-8-sig") as f:
        json.dump(report, f, indent=4, ensure_ascii=False, sort_keys=False)
        print("save file")
