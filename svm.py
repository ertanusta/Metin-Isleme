# Linear Support Vector Machine
# %92 başarı
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import pickle
data = load_files(r"C:\Users\ertan\Desktop\Metin-Isleme-master\stemmer",encoding="utf-8")
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sgd = Pipeline([('vect', TfidfVectorizer(max_features=9000, min_df=7, max_df=0.8)),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)

new_test_data=["Ekrem İmamoğlu’nun mazbatayı aldıktan sadece 1 hafta sonra projelendirip tüm inşaat çalışmalarını bitirip hizmete açtığı MARMARAY eserini kullandım.Çok güzel bir iş olmuş.Bu büyük hizmeti kısa sürede başardığınız için teşekkürler başkan"]

y_pred = sgd.predict(X_test)
tags=['ekonomi','kültür Sanat','magazin','sağlık','siyaset','spor','teknoloji']
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=tags))
with open('linear', 'wb') as picklefile:
    pickle.dump(sgd,picklefile)

with open('linear', 'rb') as training_model:
    model = pickle.load(training_model)
y_pred2 = model.predict(new_test_data)
print(model.predict_proba(new_test_data))

