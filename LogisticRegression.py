# Logistic Regression
# %93.3 başarı

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
import pickle
data = load_files(r"C:\Users\ertan\Desktop\Metin-Isleme-master\stemmer",encoding="utf-8")
X, y = data.data, data.target
tags=['ekonomi','kültür Sanat','magazin','sağlık','siyaset','spor','teknoloji']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
logreg = Pipeline([('vect', TfidfVectorizer(max_features=3600, min_df=3, max_df=0.7)),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

with open('logistic', 'wb') as picklefile:
    pickle.dump(logreg,picklefile)

with open('logistic', 'rb') as training_model:
    model = pickle.load(training_model)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=tags))