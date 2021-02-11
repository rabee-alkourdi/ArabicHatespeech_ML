# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_excel(r'../initial datasets/labeledDataset_EN_N.xlsx', encoding = 'utf-8',  index_col=0); 

data = data.sample(frac=1, random_state=42)
data['text'] = data['text'].astype(str)
data['class'] = data['class'].astype(str)

# Encode labels (strings -> numbers)
encoder = LabelEncoder()
data['class'] = encoder.fit_transform(data['class'])

df = data[['text', 'class']] 

data = df.loc[(df['class'] == 0) | (df['class'] == 2)]
data.loc[df['class'] == 2, 'class'] = 1

positive_class = data.loc[data['class'] == 0]
negative_class = data.loc[data['class'] == 1]
negative_class_portion = negative_class.sample(n = positive_class.shape[0])
train = pd.concat([positive_class, negative_class_portion])
train = train.sample(frac=1, random_state=42)

train.to_excel('../feature datasets/en/labels.xlsx', index_label="index",encoding = 'ISO-8859-1')