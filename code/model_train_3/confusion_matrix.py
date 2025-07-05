import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

CAMP_TYPE = "Type1"
CURRENT_MODEL="BERT"
DATA_TYPE="Origin"


data = np.array([(16, 3, 16), (3, 4, 28), (4, 9, 621)])
labels = ['Negative', 'Neutral', 'Positive']

# Create the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(data, annot=True, cmap='YlGnBu', fmt='d', linewidths=0.1, xticklabels=labels, yticklabels=labels)

plt.ylabel('Actual sentiment')
plt.xlabel('Predict sentiment')
plt.title(f"BERT Confusion Matrix(Batch Size = 8)")
plt.savefig(fname=f"{CAMP_TYPE}_{DATA_TYPE}_8_{CURRENT_MODEL}.jpg")

plt.close()


data = np.array([(22, 7, 6), (5, 11, 19), (10, 33, 591)])
plt.figure(figsize=(6, 5))
sns.heatmap(data, annot=True, cmap='YlGnBu', fmt='d', linewidths=0.1, xticklabels=labels, yticklabels=labels)

plt.ylabel('Actual sentiment')
plt.xlabel('Predict sentiment')
plt.title(f"BERT Confusion Matrix(Batch Size = 16)")
plt.savefig(fname=f"{CAMP_TYPE}_{DATA_TYPE}_16_{CURRENT_MODEL}.jpg")



# 打開兩張圖片
img1 = Image.open(f"{CAMP_TYPE}_{DATA_TYPE}_8_{CURRENT_MODEL}.jpg")
img2 = Image.open(f"{CAMP_TYPE}_{DATA_TYPE}_16_{CURRENT_MODEL}.jpg")

# 確保兩張圖片尺寸相同（可選擇縮放）
img2 = img2.resize(img1.size)

# 水平合併（side by side）
merged_h = Image.new("RGB", (img1.width + img2.width, img1.height))
merged_h.paste(img1, (0, 0))
merged_h.paste(img2, (img1.width, 0))
merged_h.save(f"{CAMP_TYPE}_{DATA_TYPE}_{CURRENT_MODEL}.jpg")