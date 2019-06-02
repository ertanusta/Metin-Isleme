# TR: 2.TFxIdf Hesaplama Adımları
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)
path="stemmer/ekonomi/"
all_files=os.listdir(path)
docs=[]
i=0
while(len(all_files)>i):
    f = open(path + str(all_files[i]), "r", encoding='utf8')
    text = f.read()
    docs.append(text)
    i = i + 1
Tfidf_Vector = TfidfVectorizer(min_df = 0, max_df = 100, use_idf=True)
Tfidf_Matrix = Tfidf_Vector.fit_transform(docs)
print(Tfidf_Matrix)
Tfidf_Matrixx = Tfidf_Matrix.toarray()
print(Tfidf_Matrixx)
# TR: Tfidf_Vector içerisindeki tüm öznitelikleri al
features = Tfidf_Vector.get_feature_names()
print(features)
# TR: Doküman - öznitelik matrisini göster
Tfidf_df = pd.DataFrame(np.round(Tfidf_Matrixx, 3), columns = features)
#w = open(path + "/tumsonuc.txt", "w", encoding='utf8')
#w.write(str(Tfidf_df))


#print(Tfidf_df)
print("----")


