import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

CAMP_TYPE = "Type2"
CURRENT_MODEL="BERT"  # MultilingualBERT # DistilBERT # RoBERTa # ALBERT # BERT
DATA_TYPE="GPT-4o mini" # GPT-4o mini
data = np.array([(32, 0, 9), (2, 6, 10), (5, 10, 562)])


labels = ['Negative', 'Neutral', 'Positive']

# Create the heatmap
matplotlib.rc('font', family='Microsoft JhengHei')
plt.figure(figsize=(6, 5))
sns.heatmap(data, annot=True, cmap='YlGnBu', fmt='d', linewidths=0.1, xticklabels=labels, yticklabels=labels)

plt.ylabel('Actual sentiment')
plt.xlabel('Predict sentiment')
plt.title(f"{CURRENT_MODEL} Confusion Matrix(Batch Size = 8)\n(使用{DATA_TYPE}做資料增生)")
# plt.title(f"{CURRENT_MODEL} Confusion Matrix(Batch Size = 8)")
plt.savefig(fname=f"{DATA_TYPE}\\{CAMP_TYPE}\\8_{CURRENT_MODEL}.jpg")

plt.close()


data = np.array([(28, 0, 13), (0, 7, 11), (2, 4, 571)])
plt.figure(figsize=(6, 5))
sns.heatmap(data, annot=True, cmap='YlGnBu', fmt='d', linewidths=0.1, xticklabels=labels, yticklabels=labels)

plt.ylabel('Actual sentiment')
plt.xlabel('Predict sentiment')
plt.title(f"{CURRENT_MODEL} Confusion Matrix(Batch Size = 16)\n(使用{DATA_TYPE}做資料增生)")
# plt.title(f"{CURRENT_MODEL} Confusion Matrix(Batch Size = 16)")

plt.savefig(fname=f"{DATA_TYPE}\\{CAMP_TYPE}\\16_{CURRENT_MODEL}.jpg")



# 打開兩張圖片
img1 = Image.open(f"{DATA_TYPE}\\{CAMP_TYPE}\\8_{CURRENT_MODEL}.jpg")
img2 = Image.open(f"{DATA_TYPE}\\{CAMP_TYPE}\\16_{CURRENT_MODEL}.jpg")

# 確保兩張圖片尺寸相同（可選擇縮放）
img2 = img2.resize(img1.size)

# 水平合併（side by side）
merged_h = Image.new("RGB", (img1.width + img2.width, img1.height))
merged_h.paste(img1, (0, 0))
merged_h.paste(img2, (img1.width, 0))
merged_h.save(f"{DATA_TYPE}\\{CAMP_TYPE}\\{CURRENT_MODEL}.jpg")