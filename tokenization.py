import nltk
import numpy as np
import re
import pandas as pd
import os
pd.options.display.max_colwidth = 8000

#
#Tüm veri seti içerisinde temizleme yaparak yeni bir .txt oluşturur.
#Stop word iyi çalışıyor sadece noktolama işarelerini de eklemelisin
#
stop_word_list=[]
f = open("stopwords_tr.txt", "r", encoding='utf8')
for i in f:
    stop_word_list.append(re.sub('\n', '', i))
f.close()

def norm_doc(single_doc):
    #belirlenen özel karakterleri ve sayıları temizle
    single_doc = re.sub(" \d+", " ", single_doc)
    single_doc = re.sub("'", " ", single_doc)
    single_doc = re.sub('"', " ", single_doc)
    single_doc = re.sub('“', " ", single_doc)
    single_doc = re.sub('”', " ", single_doc)
    single_doc = re.sub('’', " ", single_doc)
    # küçük harflere çevir
    single_doc = single_doc.lower()
    single_doc = single_doc.strip()
    # token'larına ayır
    tokens = WPT.tokenize(single_doc)
    # Stop-word listesindeki kelimeler hariç al
    filtered_tokens = [token for token in tokens if token not in stop_word_list]
    # Dokümanı tekrar oluştur
    single_doc = ' '.join(filtered_tokens)
    return single_doc
i=0

path="TTC-3600_Orj/spor/"
all_files=os.listdir(path)
while(len(all_files)>i):
    print (all_files[i])
    f = open(path + str(all_files[i]), "r", encoding='utf-8')
    text = f.read()
    #İ harfinde bir problem olduğundan böyle bir çözüme gidildi
    text=text.replace("İ","i")
    text=text.replace("Ç","ç")
    text = text.replace("Ö", "ö")
    text = text.replace("Ğ", "ğ")
    text = text.replace("Ş", "ş")
    docs = np.array(text)
    WPT = nltk.WordPunctTokenizer()
    norm_docs = np.vectorize(norm_doc)
    normalized_documents = norm_docs(docs)
    w=open("tokenization/spor/"+str(all_files[i]),"w",encoding='utf-8')
    w.write(str(normalized_documents))
    i =i+ 1
    f.close()
    w.close()

