# pip install transformers>=4.34
# pip install accelerate

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import jieba


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

messages = [
    {
        "role": "role", 
        "content": """你是一個經營露營地的人員，露營地名稱叫做「蟬說：山中靜靜」，請你根據貴賓的評論給予回覆，如果貴賓給予的是正面評價，請你回應貴賓感激的話，如果貴賓回覆的是負面評價，請你根據貴賓提及的問題，先道歉讓貴賓感受到誠意，接著提出未來改善的方向給貴賓。若你無法回答，請你說「很抱歉，我無法回答您的問題。」"""
    },
    {
        "role": "assistant", 
        "content": "好的，請您上傳露營的評論內容。"
    },
    {
        "role": "user", 
        "content": "如果園區是需要牽繩的請不要用這種讓飼主誤會可以放繩的廣告照片在官網上跟FB /IG 上猛打廣告做不到讓狗狗跑跑寵物友善請不要勉強來之前很興奮來之後很心酸10點後需放低音量的規則園區也無人管理現在00:49隔壁帳的人還開手機擴音在尬聊"
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
     