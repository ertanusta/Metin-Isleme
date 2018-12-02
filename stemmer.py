import snowballstemmer
kok=snowballstemmer.TurkishStemmer()
import os
path="tokenization/siyaset/"
all_files=os.listdir(path)
i=0
text=""
while(len(all_files)>i):
    f = open(path + str(all_files[i]), "r", encoding='utf8')
    text = f.read()
    new_text=kok.stemWords(text.split())
    w = open("stemmer/siyaset/" + str(all_files[i]), "w", encoding='utf8')
    w.write(str(new_text))
    i = i + 1