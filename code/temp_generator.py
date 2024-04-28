from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = "shenzhi-wang/Llama3-8B-Chinese-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

torch.save(model.state_dict(), "Llama3_8B_Chinese.pt")

messages = [
    {
        "role": "system",
        "content": "請幫我做資料增生",
    },
    {"role": "user", "content": "草地很棒, 浴室乾淨水壓夠! 營主熱情又熱心, 還會安排採咖啡豆活動讓小朋友玩! 會再訪的營區!\n\n 以上評論請使用繁體中文來換句話說，盡量表達出句中所提到每一個重點，照著這個規則請列出類似的評論"},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=8196,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(response)
print(tokenizer.decode(response, skip_special_tokens=True))
