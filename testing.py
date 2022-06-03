import json
from re import S
from keras import optimizers
from keras.optimizer_v1 import Adam, Optimizer
from numpy.core.fromnumeric import ndim
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import string

token = Tokenizer()

print('Load Model')
load_model = keras.models.load_model('modelAnalisis')
print(load_model)
print('Compile model')
load_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('Done')

print('Model : ', load_model)

x = 'Awsome'

data = token.texts_to_sequences(x)
data = pad_sequences(data, maxlen= 100, padding='post')

prediksiData = load_model.predict(data)
prediksi = np.argmax(prediksiData, axis=1)

print('Predict Model : ', prediksi)