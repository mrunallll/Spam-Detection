# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:17:26 2020

@author: Admin
"""

#Approach 1 #NLP through csv dataset, lemmatization

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
import pickle


data= pd.read_csv('emails1.csv')
data.head(5)

#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

corpus=[]
for i in range(0, 5728):
  mail=re.sub('[^a-zA-Z]', ' ', data['text'][i])
  mail.lower()
  mail.split()
  for word in mail:
    word= lem.lemmatize(word)
  mail.split()
  corpus.append(mail)
#print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=55000)
X= cv.fit_transform(corpus).toarray()
y=data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

#Fitting SVMs to our dataset
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
#MAking confusion MAtrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=2)

pickle.dump(classifier, open('nlp_model.pkl', 'wb'))
pickle.dump(cv, open('transform.pkl', 'wb'))