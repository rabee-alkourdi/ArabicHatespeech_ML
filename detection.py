# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from feature_engineering.count_features import count_features
from feature_engineering.tfidf_features import tfidf_features
from feature_engineering.word_bigrams_features import word_bigrams_features
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from pprint import pprint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

    
class_labels = pd.read_excel('feature datasets/en/cleaned_data_stemmed.xlsx', encoding='utf-8', index_col=0)
pos_features = pd.read_excel('feature datasets/en/pos_features.xlsx', encoding='utf-8', index_col=0)
deprel_features = pd.read_excel('feature datasets/en/deprel_features.xlsx', encoding='utf-8', index_col=0)
tfidf_sparse_matrix = tfidf_features('en', 'clean_stemmed', 5000)
count_sparse_matrix = count_features('en', 'clean', 1000)
word_bigrams = word_bigrams_features("clean", 1000)

#merge all feature data sets based on 'index' column sentiment_scores, dependency_features, char_bigrams, word_bigrams
df_list=[class_labels, tfidf_sparse_matrix]
master = df_list[0]
for df in df_list[1:]:
    master = master.merge(df, on='index')

master.columns.values
#master = master.sample(frac=0.4, random_state=42)


#ignore first two columns (index and tweet)
y=master.iloc[:,1] #class labels
X=master.iloc[:,2:] #all features

#create train and test sets: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#from sklearn.cluster import KMeans
#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#print(purity_score(y, kmeans.labels_))
clf = LinearSVC()
#clf = LogisticRegression(solver='lbfgs')
#clf = GradientBoostingClassifier(n_estimators=500, learning_rate=.025)


scorer = {'precision', 'recall', 'f1', 'accuracy'}

scores = cross_validate(clf, X_train, y_train, scoring=scorer, cv=10)


avgScores = {}
for k,v in scores.items():
    avgScores[k] = round(sum(v)/ float(len(v)), 4)
pprint(avgScores)


# Confusion Matrix

from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
#print ('fscore:{0:.3f}'.format(f1_score(y_test, pred)))
from sklearn.metrics import confusion_matrix
confusion_lr = confusion_matrix(y_test, pred)

# ROC

import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, pred)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();
'''
from keras import layers, models, optimizers

def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))
    
    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print "CNN, Word Embeddings",  accuracy
'''