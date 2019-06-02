# Keras
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import sklearn.pipeline as Pipeline
from keras.models import model_from_json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

data = load_files(r"C:\Users\ertan\Desktop\Metin-Isleme-master\stemmer",encoding="utf-8")
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
tfidf = TfidfVectorizer(binary=True,max_features=9000, min_df=7, max_df=0.8)
X = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
lb = LabelEncoder()
y = lb.fit_transform(y_train)
y_test=lb.fit_transform(y_test)
y_test=np_utils.to_categorical(y_test)
dummy_y_train = np_utils.to_categorical(y)
print(X.shape,dummy_y_train.shape)
# Model Training
print ("Create model ... ")

model = Sequential()
model.add(Dense(512, input_dim=9000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(160, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X, dummy_y_train, epochs = 24, batch_size = 128, validation_data=(X_test, y_test))



print ("Predict on test data ... ")
y_test = model.predict(X_test)
print(y_test)
# y_pred = lb.inverse_transform(y_test)

score, acc = model.evaluate(X_test, y_test,
                            batch_size=128)

print(score)
print(acc)


model.save_weights('model_weights.h5')
pickle.dump(tfidf, open("vectorizer.pickle", "wb"))
# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights('model_weights.h5')

vectorizer = pickle.load(open("vectorizer.pickle","rb"))

new_test_data=["Ekrem İmamoğlu’nun mazbatayı aldıktan sadece 1 hafta sonra projelendirip tüm inşaat çalışmalarını bitirip hizmete açtığı MARMARAY eserini kullandım.Çok güzel bir iş olmuş.Bu büyük hizmeti kısa sürede başardığınız için teşekkürler başkan"]

test=vectorizer.transform(new_test_data)
print("---")
print(test)
result=model.predict(test)
for i in result[0]:
    print(round(float(i),3))

print(model.predict_proba(test,batch_size=128,verbose=1))
print(model.predict_classes(test))