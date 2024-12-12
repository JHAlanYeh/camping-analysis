import requests
import json


user_id = "U30bc9aaf24ea900745f69d036821e5e3"
channel_access_token = "kQ/vmM5sqBwC9Dyc4ODf1aD/CeTTHZKbU32UCZsxHKgsIglO7oqC29E6mgJcU8jpfT57f2Ordq0fglHqXXw33D5142fPiE6mCCT1m5PJ38Jnfu2qBhXp2m6q01jAZBI4LSc+qmSZzUze5AVUN4obEwdB04t89/1O/w1cDnyilFU="


headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + channel_access_token
}
push_text =  "2024 49週報告已產出，請至後台查看。"
print(push_text)
res = requests.post("https://api.line.me/v2/bot/message/push", json={"to": user_id, "messages": [{"type": "text", "text": json.dumps(push_text)}]}, headers=headers)
print(res.text)