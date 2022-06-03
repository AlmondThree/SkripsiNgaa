from __future__ import print_function
import flask
from nltk import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from werkzeug.utils import secure_filename
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
import pandas as pd
from tensorflow.python.keras import backend as K
from matplotlib import pyplot
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPool1D, Embedding
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show
import numpy as np
import sys
from flask import Flask, app, redirect, url_for, render_template, request, session, flash, make_response
from datetime import timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import string
import re
from io import BytesIO

#NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import tokenize
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.sentiment.util import *

token = Tokenizer()

#sci-kit learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Setup template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'templates')

UPLOAD_FOLDER = 'templates/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder= TEMPLATE_DIR)

#mMelakukan set routing web
@app.route('/home',methods=['GET', 'POST'])
def index() :
    return render_template('index.html')

@app.route('/coba',methods=['GET', 'POST'])
def coba() :
    nama = request.form['nama']
    model = loaded_model
    return render_template('coba.html', nama=nama, model=model)

@app.route('/test',methods=['GET', 'POST'])
def test() :
    if request.method=='POST':
        review = request.form['review']
        review_encode = get_encode(review)
        result = get_predict(review_encode)
        return render_template('test.html', review=result)
    return render_template('test.html')
"""
print("Loading model ....")
#Load json and create model
json_file = open("taruh jsonnya disini")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
"""
loaded_model = keras.models.load_model('modelAnalisis')
#Load weights into new model
#loaded_model.load_weights("modelAnalisis")
#print("Loaded model from disk")
#Evaluate loaded model on test data
print('compiling model')
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print('done')
print("Model : ", loaded_model)

def get_encode(x):
    x = token.texts_to_sequences(x)
    x = pad_sequences(x, maxlen= 100, padding='post')
    return x

def get_predict(x):
    model = loaded_model
    result1 = model.predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
    result = np.argmax(result1, axis=1)
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0')