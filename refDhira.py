import pandas as pd
import tqdm
import numpy as np
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
from tqdm import tqdm, notebook
import pandas as pd
from typing import List
import time
import asyncio
import keras.backend as K
from collections import Counter
from datetime import datetime
 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid')
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Dense,
    Dropout,
    Reshape,
    Flatten,
    concatenate,
    Input,
    Conv1D,
    GlobalMaxPooling1D,
    Embedding,
    LSTM,
    MaxPooling1D
)
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import collections
import re
import string
import nest_asyncio
import twint

c = twint.Config()
c.Search = '"sinovac" lang:id'
c.Since = "2021-05-25"
c.Until = "2021-06-18"
c.Store_csv = True
c.Output = 'tweets_data.csv'
tweets_data = pd.read_csv('/content/gdrive/My Drive/tweets_data.csv') #untuk baca file csv
tweets = tweets_data[['id', 'username', 'created_at', 'tweet']] #buat nampilin data frame
tweets

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers

    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
    return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower() 
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text) 
    return text

def filteringText(text): # Remove stopwords in a text
    listStopwords = set(stopwords.words('indonesian'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered 
    return text

def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence


# # Preprocessing tweets data

tweets['text_clean'] = tweets['tweet'].apply(cleaningText)
tweets['text_clean'] = tweets['text_clean'].apply(casefoldingText)

tweets['text_preprocessed'] = tweets['text_clean'].apply(tokenizingText)
tweets['text_preprocessed'] = tweets['text_preprocessed'].apply(filteringText)
tweets['text_preprocessed'] = tweets['text_preprocessed'].apply(stemmingText)

tweets = pd.read_csv('/content/gdrive/My Drive/tweets_data_clean.csv')

for i, text in enumerate(tweets['text_preprocessed']):
    tweets['text_preprocessed'][i] = tweets['text_preprocessed'][i].replace("'", "")\
                                            .replace(',','').replace(']','').replace('[','')
    list_words=[]
    for word in tweets['text_preprocessed'][i].split():
        list_words.append(word)
        
    tweets['text_preprocessed'][i] = list_words  
tweets

positif = dict()
import csv
with open("/content/gdrive/My Drive/pos.txt") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    for row in reader:
          positif[row[0]] = int(row[1])
        
negatif = dict()
import csv
with open('/content/gdrive/My Drive/neg.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    for row in reader:
        negatif[row[0]] = int(row[1])
        
# Function to determine sentiment polarity of tweets        
def sentiment_analysis(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in positif):
            score = score + positif[word]
    for word in text:
        if (word in negatif):
            score = score + negatif[word]
    polarity=''
    if (score > 0):
        polarity = 'positive'
    elif (score < 0):
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return np.asarray([score, polarity])

# Results from determine sentiment polarity of tweets

results = tweets['text_preprocessed'].apply(sentiment_analysis)
results = list(zip(*results))
tweets['polarity_score'] = results[0]
tweets['polarity'] = results[1]
print(tweets['polarity'].value_counts())

# Export to csv file
tweets.to_csv(r'/content/gdrive/My Drive/polarity_data.csv', index = False, header = True,index_label=None)
tweets

# Pie Chart
fig, ax = plt.subplots(figsize = (6, 6))
sizes = [count for count in tweets['polarity'].value_counts()]
labels = list(tweets['polarity'].value_counts().index)
explode = (0.1, 0, 0)
ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', 
       explode = explode, textprops={'fontsize': 14})
plt.savefig("/content/gdrive/My Drive/pie_chart.png")

def words_with_sentiment(text):
    positive_words=[]
    negative_words=[]
    neutral_words=[]
    for word in text:
        score_pos = 0
        score_neg = 0
        if (word in positif):
            score_pos = positif[word]
        if (word in negatif):
            score_neg = negatif[word]
        
        if (score_pos + score_neg > 0):
            positive_words.append(word)
        elif (score_pos + score_neg < 0):
            negative_words.append(word)
        else:
            neutral_words.append(word)

    return positive_words, negative_words,neutral_words

# Make text preprocessed (tokenized) to untokenized with toSentence Function
X = tweets['text_preprocessed'].apply(toSentence) 
max_features = 5000

# Tokenize text with specific maximum number of words to keep, based on word frequency
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(X.values)
X = tokenizer.texts_to_sequences(X.values)
X = pad_sequences(X)
X.shape

# Encode target data into numerical values
polarity_encode = {'negative' : 0, 'neutral' : 1, 'positive' : 2}
y = tweets['polarity'].map(polarity_encode).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from keras.layers import Embedding, Dense, Dropout, LSTM
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load Word2Vec Indonesia
import gensim
from gensim import models
word2vec = '/content/gdrive/My Drive/idwiki_word2vec_200.model'
id_w2v = gensim.models.word2vec.Word2Vec.load(word2vec)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [
            vector[word] if word in vector else np.random.rand(k)
            for word in tokens_list
        ]
    else:
        vectorized = [
            vector[word] if word in vector else np.zeros(k) for word in tokens_list
        ]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments["text_preprocessed"].apply(
        lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing)
    )
    return list(embeddings)

# Split data into test and train
df = pd.read_csv(filepath_or_buffer='/content/gdrive/My Drive/polarity_data.csv', nrows=500)
data_train, data_test = train_test_split(df, test_size=0.25, random_state=42)
print('Training Data:',len(data_train))
print('Testing Data:',len(data_test))
data_train.to_csv(r"/content/gdrive/My Drive/datatrain1.csv")
data_test.to_csv(r"/content/gdrive/My Drive/datatest1.csv")

# Make Dictionary Data Train
    # all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
all_training_words = [
        word for tokens in data_train["text_preprocessed"] for word in eval(tokens)
    ]
training_sentence_lengths = [len(eval(tokens)) for tokens in data_train["text_preprocessed"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print(
        "%s words total, with a vocabulary size of %s"
        % (len(all_training_words), len(TRAINING_VOCAB))
    )
print("Max sentence length is %s" % max(training_sentence_lengths))

    # Make Dictionary Data Test
all_test_words = [word for tokens in data_test["text_preprocessed"] for word in eval(tokens)]
test_sentence_lengths = [len(eval(tokens)) for tokens in data_test["text_preprocessed"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print(
        "%s words total, with a vocabulary size of %s"
        % (len(all_test_words), len(TEST_VOCAB))
    )
print("Max sentence length is %s" % max(test_sentence_lengths))

# Projected Token To Vector Using Word2Vec
training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)
import math

# MAX_SEQUENCE_LENGTH = 50 menyamakan panjang twitter
import math
MAX_SEQUENCE_LENGTH = max(training_sentence_lengths)
MAX_SEQUENCE_LENGTH = int(math.ceil((MAX_SEQUENCE_LENGTH) / 10.0)) * 10
EMBEDDING_DIM = 100
### Tokenize and Add Padding sequences
tokenizer = Tokenizer(
        num_words=len(TRAINING_VOCAB),
        lower=True,
        char_level=False,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    )
tokenizer.fit_on_texts(data_train["text_preprocessed"].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train["text_preprocessed"].tolist())
print("training_sequences(MAX) : ", max(training_sequences))

# Tokenize and Pad sequences

tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["text_preprocessed"].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train["text_preprocessed"].tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))
print(training_sequences)

dict={'token': training_sequences}
df = pd.DataFrame(dict) 
df.to_csv(r"/content/gdrive/My Drive/word2vec.csv")

train_word_index = tokenizer.word_index
print("Found %s unique tokens." % len(train_word_index))

train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("train_cnn_data size ", len(train_cnn_data))
print("train_cnn_data ", train_cnn_data)
train_embedding_weights = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))
print("train_embedding_weights size ", len(train_embedding_weights))
print("train_embedding_weights ", train_embedding_weights)

test_sequences = tokenizer.texts_to_sequences(data_test["text_preprocessed"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(test_cnn_data)
# Choose Column For Weighting Tokens
label_names = ["polarity"]
y_train = data_train[label_names].values
x_train = train_cnn_data
y_true = y_train

print("$ train_embedding_weights ", train_embedding_weights)
print("$ MAX_SEQUENCE_LENGTH ", MAX_SEQUENCE_LENGTH)
print("$ len(train_word_index) + 1 ", len(train_word_index) + 1)
print("$ EMBEDDING_DIM ", EMBEDDING_DIM)
print("$ len(list(label_names)) ", len(list(label_names)))

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2,3,4,5,6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)


    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)  
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

label_names = ['polarity']
y_train = data_train[label_names].values
x_train = train_cnn_data
y_true = y_train

model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                len(list(label_names)))

# Visualization model accuracy (train and val accuracy)

fig, ax = plt.subplots(figsize = (10, 4))
ax.plot(model_prediction.history['accuracy'], label = 'train accuracy')
ax.plot(model_prediction.history['val_accuracy'], label = 'val accuracy')
ax.set_title('Model Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(loc = 'upper left')
plt.show()

# Predict sentiment on data test by using model has been created, and then visualize a confusion matrix

y_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print('Model Accuracy on Train Data:', accuracy)
confusion_matrix(y_train, y_pred)

fig, ax = plt.subplots(figsize = (8,6))
sns.heatmap(confusion_matrix(y_true = y_train, y_pred = y_pred), fmt = 'g', annot = True)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Prediction', fontsize = 14)
ax.set_xticklabels(['negative (0)', 'neutral (1)', 'positive (2)'])
ax.set_ylabel('Actual', fontsize = 14)
ax.set_yticklabels(['negative (0)', 'neutral (1)', 'positive (2)'])
plt.show()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy on Test Data:', accuracy)
confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize = (8,6))
sns.heatmap(confusion_matrix(y_true = y_test, y_pred = y_pred), fmt = 'g', annot = True)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Prediction', fontsize = 14)
ax.set_xticklabels(['negative (0)', 'neutral (1)', 'positive (2)'])
ax.set_ylabel('Actual', fontsize = 14)
ax.set_yticklabels(['negative (0)', 'neutral (1)', 'positive (2)'])
plt.show()

# Wordcloud
from wordcloud import WordCloud, STOPWORDS

comment_words = ""
stopwords = set(STOPWORDS)
comment_words += " ".join(df.text_preprocessed) + " "
print(len(comment_words))
wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        stopwords=stopwords,
        min_font_size=10,
        collocations=False,
    ).generate(comment_words)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("/content/gdrive/My Drive/graph_word_cloud.png")
plt.close()
