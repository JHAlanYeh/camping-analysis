import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta
import json

# ****************************************************************************** #
root_path = os.path.join(os.getcwd(), "new_data")

google_directory = "google_comments"
google_path = os.path.join(root_path, google_directory)

directorys = ["asiayo_comments", "easycamp_comments", "klook_comments"]

summary_comments_dir = os.path.join(root_path, "summary_comments")


for file in os.listdir(google_path):
    comments = []
    google_file_path = os.path.join(google_path, file)
    print(file)

    gf = open(google_file_path, encoding="utf-8-sig")
    google_data = json.load(gf)
    gf.close()

    comments = comments + google_data["comments"]

    for dir in directorys:
        target_dir = os.path.join(root_path, dir)
        target_file_path = os.path.join(target_dir, file)
        if os.path.exists(target_file_path):
            tf = open(target_file_path, encoding="utf-8-sig")
            target_data = json.load(tf)
            tf.close()
            comments = comments + target_data["comments"]
    print(len(comments))
    print("====================================")


    target_file_path = os.path.join(summary_comments_dir, file)
    with open(target_file_path, 'w', encoding="utf-8-sig") as nf:
        json.dump(comments, nf, indent=4, ensure_ascii=False)
        print("save {file_name}".format(file_name=target_file_path))