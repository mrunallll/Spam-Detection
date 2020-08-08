# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 17:26:28 2020

@author: Admin
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
#from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app=Flask(__name__)
classifier=pickle.load(open('nlp_model.pkl', 'rb'))
cv=pickle.load(open('transform.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data= [message]
        vect=cv.transform(data).toarray()
        my_pred=classifier.predict(vect)
    return render_template('result.html', prediction= my_pred)

if __name__=="__main__":
    app.run(debug=True)