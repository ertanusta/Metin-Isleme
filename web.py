#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import re
import sys
import pickle
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
import pickle
from keras.models import model_from_json

stop_word_list = ["﻿er", "acaba", "altmış", "altı", "ama", "ilk", "ancak", "arada", "aslında", "ayrıca", "bana",
                  "bazı", "belki", "ben", "benden", "beni", "benim", "beri", "beş", "bile", "bin", "bir", "birçok",
                  "biri", "birkaç", "birkez", "birşey", "birşeyi", "biz", "bize", "bizden", "bizi", "bizim", "böyle",
                  "böylece", "bu", "buna", "bunda", "bundan", "bunlar", "bunları", "bunların", "bunu", "bunun",
                  "burada",
                  "çok", "çünkü", "da", "daha", "dahi", "de", "defa", "değil", "diğer", "diye", "doksan", "dokuz",
                  "dolayı",
                  "dolayısıyla", "dört", "edecek", "eden", "ederek", "edilecek", "ediliyor", "edilmesi", "ediyor",
                  "eğer",
                  "elli", "en", "etmesi", "etti", "ettiği", "ettiğini", "gibi", "göre", "halen", "hangi", "hatta",
                  "hem", "henüz",
                  "hep", "hepsi", "her", "herhangi", "herkesin", "hiç", "hiçbir", "için", "iki", "ile", "ilgili", "ise",
                  "işte",
                  "itibaren", "itibariyle", "kadar", "karşın", "katrilyon", "kendi", "kendilerine", "kendini",
                  "kendisi",
                  "kendisine", "kendisini", "kez", "ki", "kim", "kimden", "kime", "kimi", "kimse", "kırk", "milyar",
                  "milyon",
                  "mu", "mü", "mı", "nasıl", "ne", "neden", "nedenle", "nerde", "nerede", "nereye", "niye", "niçin",
                  "o", "olan",
                  "olarak", "oldu", "olduğu", "olduğunu", "olduklarını", "olmadı", "olmadığı", "olmak", "olması",
                  "olmayan",
                  "olmaz", "olsa", "olsun", "olup", "olur", "olursa", "oluyor", "on", "ona", "ondan", "onlar",
                  "onlardan", "onları",
                  "onların", "onu", "onun", "otuz", "oysa", "öyle", "pek", "rağmen", "sadece", "sanki", "sekiz",
                  "seksen", "sen",
                  "senden", "seni", "senin", "siz", "sizden", "sizi", "sizin", "şey", "şeyden", "şeyi", "şeyler",
                  "şöyle", "şu",
                  "şuna", "şunda", "şundan", "şunları", "şunu", "tarafından", "trilyon", "tüm", "üç", "üzere", "var",
                  "vardı", "ve",
                  "veya", "ya", "yani", "yapacak", "yapılan", "yapılması", "yapıyor", "yapmak", "yaptı", "yaptığı",
                  "yaptığını",
                  "yaptıkları", "yedi", "yerine", "yetmiş", "yine", "yirmi", "yoksa", "yüz", "zaten", "1", "2", "3",
                  "4", "5", "6", "7"
    , "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26",
                  "27", "28", "29", "30"
    , "31", "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım",
                  "aralık", "hafta",
                  "ay", "gün", "saat", ":", ",", ";", "!", "?", "-", "_", "/", "*", "+", "(", ")", "{", "}", "%", "&",
                  "#", '"', "'", "@", "."]


def norm_doc(single_doc):
    single_doc = re.sub(" \d+", " ", single_doc)

    single_doc = single_doc.lower()
    single_doc = single_doc.strip()

    single_doc = single_doc.split(" ")

    filtered_tokens = [token for token in single_doc if token not in stop_word_list]

    single_doc = ' '.join(filtered_tokens)
    return single_doc



modelSelect = sys.argv[1]
text = ""
f = open("deneme.txt", "r", encoding='utf8')
for i in f:
    text = text + i + " "
f.close()
text = text.replace("İ", "i")
text = text.replace("Ç", "ç")
text = text.replace("Ö", "ö")
text = text.replace("Ğ", "ğ")
text = text.replace("Ş", "ş")
docs = np.array(text)

norm_docs = np.vectorize(norm_doc)
normalized_documents = norm_docs(docs)
# 0=>RandomForest
# 1=>Naive Bayes Classifier
# 2=>Linear Support Vector Machine
# 3=>Logistic Regression
if (modelSelect == "0"):
    with open('keras.json', 'rb') as f:
        model = model_from_json(f.read())
    model.load_weights('keras.h5')
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    test = vectorizer.transform([str(normalized_documents)])
    result = model.predict(test)
    for i in result[0]:
        print(round(float(i), 3))

elif (modelSelect == "1"):
    with open('/var/www/html/Faust/public/script/bayes', 'rb') as training_model:
        model = pickle.load(training_model)
    result = model.predict_proba([str(normalized_documents)])
    for i in result[0]:
        print(round(i * 100, 2), ",")
elif (modelSelect == "2"):
    with open('/var/www/html/Faust/public/script/linear', 'rb') as training_model:
        model = pickle.load(training_model)
    result = model.predict_proba([str(normalized_documents)])
    for i in result[0]:
        print(round(i * 100, 2), ",")
elif (modelSelect == "3"):
    with open('/var/www/html/Faust/public/script/logistic', 'rb') as training_model:
        model = pickle.load(training_model)
    result = model.predict_proba([str(normalized_documents)])
    for i in result[0]:
        print(round(i * 100, 2), ",")

