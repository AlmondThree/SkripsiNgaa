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

from tensorflow.python.keras.layers.pooling import GlobalMaxPool1D, GlobalMaxPooling1D, MaxPooling1D

df = pd.read_csv('review.csv')

#punctuation removal
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["ulasan_"] = df['ulasan_'].apply(lambda text: remove_punctuation(text))
#df.to_csv("flip_preprocessing_punctuationremoval.csv", sep=";", index = 0)

df.head()

puncts = [',', '.', '"', ':', '(', ')', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '<', '%', '=', '#', '*', '+', '//', '\\', '@', '~', '`', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, '')
    return x

df['ulasan_'] = df['ulasan_'].apply(lambda text: remove_punctuation(text))

df.head()

#Case Folding
df['ulasan_'] = df['ulasan_'].str.lower()

df.head(20)

#stopwords removal
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StopWordRemoverFactory()

stopwords = ['yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua', 'ia', 'seperti', 'jika', 'sehingga', 'oleh']

def remove_stopwords(text_remove) :
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text_remove).split() if word not in stopwords])

df['ulasan_'] = df['ulasan_'].apply(lambda text: remove_stopwords(text))

#Stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()

stemmer = factory.create_stemmer()

def stemming(text):
    return stemmer.stem(text)

df['ulasan_'] = df['ulasan_'].apply(lambda text: stemming(text))

y = df['Sentimen']
y = to_categorical(y)

text = df['ulasan_'].tolist()

#Tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
#print('Done')
#print(tokenizer.index_word)
vocab = len(tokenizer.index_word)+1
max_kata = 100
x = pad_sequences(sequences, maxlen=max_kata, padding='post')

#Split train and test Data
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=40, test_size= 0.3, stratify=y)

X_train = np.array([x_train])
X_test = np.array([x_test])
Y_train = np.array([y_train])
Y_test = np.array([y_test])

""""
print("ini X Train : ", X_train)
print("ini X Test : ", X_test)
print("ini Y Train : ", Y_train)
print("ini Y Test : ", Y_test)
"""

#Embedding and Add Layer

vec_size = 300
model = Sequential()
model.add(Embedding(vocab, vec_size, input_length=max_kata))
model.add(Conv1D(64, 8, activation="relu"))
model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))
model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(3, activation='softmax'))
model.summary()


#adam

#model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['acc'])

#import time

#start = time.time()

#model.fit(X_train, Y_train, epochs=2, validation_data=(X_test, Y_test))

#print('Waktu proses: ', time.time() - start, "detik")

adam = tf.keras.optimizers.Adam(learning_rate=0.1)

model.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['acc'])

train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
valid_data = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

model.fit(train_data, 
            epochs=2,
            batch_size= 32,
            validation_data=(valid_data))

score = model.evaluate(valid_data, verbose=1)

print('Test Score : ', score[0])
print('Test Accuracy : ', score[1])

#config = model.get_config()
#json_config = model.to_json()
model.save('modelAnalisis')