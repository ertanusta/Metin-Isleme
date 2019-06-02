# Naive Bayes Classifier for Multinomial Models
# %92.3 başarı
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
import pickle

data = load_files(r"C:\Users\ertan\Desktop\Metin-Isleme-master\stemmer",encoding="utf-8")
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

nb = Pipeline([('vect', TfidfVectorizer(max_features=9000, min_df=7, max_df=0.8)),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
tags=['ekonomi','kültür Sanat','magazin','sağlık','siyaset','spor','teknoloji']
with open('bayes', 'wb') as picklefile:
    pickle.dump(nb,picklefile)

with open('bayes', 'rb') as training_model:
    model = pickle.load(training_model)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=tags))
new_test_data=["spor toto süper lig 30. hafta medipol bu akşam ev göztepe ile saat 20.30'da başla mücadele suat arslanboğa arslanboğa yardımcı serka ok ve ismail şencan"]

print(model.predict(new_test_data))