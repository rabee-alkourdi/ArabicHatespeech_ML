seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import io
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform
###############################################################################
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
###############################################################################
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))
###############################################################################
def create_embedding_matrix(filepath, word_index, embedding_dim):
    fin = io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # first line: number of words in the vocabulary and the size of the vectors
    n, d = map(int, fin.readline().split())
    
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word_index:
            idx = word_index[word]
            vector = tokens[1:]
            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
        
    return embedding_matrix

###############################################################################

data = pd.read_excel('feature datasets/en/cleaned_data_stemmed.xlsx', encoding='utf-8')
data = data.sample(frac=1, random_state=42)
data['text'] = data['text'].astype(str)

encoder = LabelEncoder()
y = encoder.fit_transform(data['class'])

X_train, X_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.2, random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


vocab_size = len(tokenizer.word_index) + 1

maxlen = 30
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embedding_dim = 300

embedding_matrix = create_embedding_matrix('initial datasets/crawl-300d-2M.vec',
                                           tokenizer.word_index,
                                           embedding_dim)
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size)

#input_dim = X_train.shape[1]  # Number of features


model = Sequential()


 
#model.add(layers.Embedding(input_dim=vocab_size,
#                           output_dim=embedding_dim,
#                           embeddings_initializer = glorot_uniform(seed=42),
#                           input_length=maxlen,
#                           trainable=True))
model.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=False))


model.add(layers.Conv1D(70, 2, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(100, 3, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(60, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras.optimizers import SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[recall, precision, f1, 'accuracy'])
model.summary()



history = model.fit(X_train, y_train, 
                    epochs=15, 
                    verbose=1, 
                    validation_split=0.2, 
                    shuffle=False,
                    batch_size=20)

#loss, recall, precision, f1, accuracy = model.evaluate(X_train, y_train, verbose=False)
#print("Training Accuracy: {:.4f}".format(accuracy))
loss, recall, precision, f1, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Recall:  {:.4f}".format(recall))
print("Testing Precision:  {:.4f}".format(precision))
print("Testing F1:  {:.4f}".format(f1))
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Testing Loss:  {:.4f}".format(loss))

'''
y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = (y_pred > 0.5)
y_y = y_pred_bool.astype(int)
from sklearn.metrics import confusion_matrix
confusion_lr = confusion_matrix(y_test, y_y)
'''
