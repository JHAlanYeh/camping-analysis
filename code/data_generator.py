import nlpaug.augmenter.word as naw

f = open('stopwords_zh_TW.dat.txt', encoding="utf-8")
STOP_WORDS = []
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))

f = open('stopwords.txt', encoding="utf-8")
lines = f.readlines()
for line in lines:
    STOP_WORDS.append(line.rstrip('\n'))


print("==================================================")
text = '年底30號到此的這批露營客素質不好，都晚上11點了，山下都還聽的到你們的好歌喉，營主不管嗎？'
print(text)

# 同義詞替換
aug = naw.SynonymAug(aug_src='wordnet', lang='cmn', stopwords=STOP_WORDS)
augmented_text = aug.augment(text)
print('Synonym Augmented Text:', augmented_text[0])

# # 隨機刪除
# aug = naw.RandomWordAug(action="delete", stopwords=STOP_WORDS)
# augmented_text = aug.augment(text)
# print("Random Delete Augmented Text:", augmented_text[0])

# # 隨機交換
# aug = naw.RandomWordAug(action="swap", stopwords=STOP_WORDS)
# augmented_text = aug.augment(text)
# print("Random Swap Augmented Text:", augmented_text[0])

# # 隨機交換
# aug = naw.RandomWordAug(action="crop", stopwords=STOP_WORDS)
# augmented_text = aug.augment(text)
# print("Random Crop Augmented Text:", augmented_text[0])
print("==================================================")
# ['eng', 'als', 'arb', 'bul', 'cmn', 'dan', 'ell', 'fin', 'fra', 'heb', 'hrv', 'isl', 'ita', 'ita_iwn', 'jpn', 'cat', 'eus', 'glg', 'spa', 
#  'ind', 'zsm', 'nld', 'nno', 'nob', 'pol', 'por', 'ron', 'lit', 'slk', 'slv', 'swe', 'tha']

# aug = naw.ContextualWordEmbsAug(stopwords=STOP_WORDS)
# augmented_text = aug.augment(text)
# print("Contextual Word Embeddings Text:", augmented_text)


from textda.data_expansion import *
print(data_expansion('生活里的惬意，无需等到春暖花开')) 