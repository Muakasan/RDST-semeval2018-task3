from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Input, Embedding, Convolution1D, Dropout, Bidirectional
from keras.models import Model
from sklearn.metrics import precision_score, f1_score, recall_score

from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import emoji
from nltk.corpus import sentiwordnet as swn

SEQ_LEN = 30
EMBED_SIZE = 300

df_given = pd.read_csv("SemEval2018-T3-train-taskA_emoji.txt", sep="\t")
y_given = np.asarray(list(df_given['Label']))

def import_embedding_matrix(tokenizer):
    w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    e2v = KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
    print("vectors loaded")
    embeddings_matrix = np.zeros((len(tokenizer.word_index)+1, EMBED_SIZE+2))
    for word, i in tokenizer.word_index.items():
        embeddings_vector = None
        try:
            embeddings_vector = w2v[word]
        except:
            try:
                embeddings_vector = e2v[word]
                print(word)
            except:
                pass
        if embeddings_vector is not None:
            ss = list(swn.senti_synsets(word))
            if ss:
                embeddings_matrix[i] = np.concatenate((embeddings_vector, np.asarray([ss[0].pos_score(), ss[0].neg_score()])))
            else:
                embeddings_matrix[i] = np.concatenate((embeddings_vector, np.asarray([0, 0])))
    return embeddings_matrix

batch_size = 32

tokenizer = Tokenizer()
texts_given = list(df_given['Tweet text'])
tokenizer.fit_on_texts(texts_given)

X_given = pad_sequences(tokenizer.texts_to_sequences(texts_given), maxlen=SEQ_LEN)

print("Tokenizer trained")    
X_train, X_test, y_train, y_test = train_test_split(X_given, y_given, random_state=42)


embedding_layer = Embedding(len(tokenizer.word_index)+1,
        EMBED_SIZE+2,
        weights=[import_embedding_matrix(tokenizer)],
        input_length=SEQ_LEN,
        trainable=False)
print("made embedding layer")

input_layer = Input(shape=(SEQ_LEN,))
embed = embedding_layer(input_layer)
first_lstm = LSTM(batch_size, return_sequences=True, dropout=.25, batch_input_shape=(batch_size, SEQ_LEN, EMBED_SIZE))(embed)
second_lstm = LSTM(batch_size, dropout=.25)(first_lstm)
first_dense = Dense(1, activation='sigmoid')(second_lstm)

model = Model(inputs=[input_layer], outputs=first_dense)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit([X_train], y_train,
          batch_size=batch_size, epochs=30, shuffle=True,
          validation_data=([X_test], y_test))

y_test_predicted = model.predict(X_test)
y_test_predicted = np.asarray([0 if x < .5 else 1 for x in y_test_predicted])

print("Precision:", precision_score(y_test, y_test_predicted))
print("Recall:", recall_score(y_test, y_test_predicted))
print("f1:", f1_score(y_test, y_test_predicted))
