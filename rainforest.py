# Rainforest
#%92.8
import numpy as np
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
new_test_data=["spor toto süper lig 30. hafta medipol bu akşam ev göztepe ile saat 20.30'da başla mücadele suat arslanboğa arslanboğa yardımcı serka ok ve ismail şencan","iç talep geliş bağ ol enflasyon gösterge bir miktar iyi bu gıda fiyat ve ithal gir maliyet art ile enflasyon beklenti yüksek seyir fiyat istikrar yönelik risk devam et bu çerçeve enflasyon görünüm belirgin bir iyi sağ kadar sıkı para dur koru kar"]
data = load_files(r"C:\Users\ertan\Desktop\Metin-Isleme-master\stemmer",encoding="utf-8")
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rfc = Pipeline([('vect', TfidfVectorizer(max_features=9000, min_df=7, max_df=0.8)),
               ('tfidf', TfidfTransformer()),
               ('clf', RandomForestClassifier(n_estimators=1000, random_state=0)),
              ])
rfc.fit(X_train, y_train)
y_pred2 = rfc.predict(X_test)


with open('randomForest', 'wb') as picklefile:
    pickle.dump(rfc,picklefile)

with open('randomForest', 'rb') as training_model:
    model = pickle.load(training_model)


print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2,target_names=['ekonomi','kültür Sanat',"magazin",'sağlık','siyaset','spor','teknoloji']))
print(accuracy_score(y_test, y_pred2))

print("---")

print(model.predict(new_test_data))
