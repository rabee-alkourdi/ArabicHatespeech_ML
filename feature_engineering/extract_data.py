# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_excel(r'../initial datasets/LabeledDataset.xlsx', sheet_name='Scenario 2'); 
data = data.sample(frac=1, random_state=42)
data['commentText'] = data['commentText'].astype(str)
data['replies.commentText'] = data['replies.commentText'].astype(str)

# Replace empty cells in commentText with np.nan
data['commentText'].replace(['nan'], np.nan, inplace=True)
# replace NaN values in commentText with values in replies.commentText
data['commentText'] = data['commentText'].fillna(data['replies.commentText'])

# Encode labels (strings -> numbers)
encoder = LabelEncoder()
data['Label'] = encoder.fit_transform(data['Label'])

# Rename columns and select text and class info
data.rename(columns = {'commentText':'text', 'Label':'class'}, inplace = True)
df = data[['text', 'class']] 

df.to_excel('../feature datasets/ar/labels.xlsx', index_label="index")