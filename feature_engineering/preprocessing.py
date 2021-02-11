# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from nltk.stem import arlstem
from nltk.corpus import stopwords
from tashaphyne import normalize

data = pd.read_excel('../feature datasets/ar/labels.xlsx', encoding = 'utf-8', index_col = 0)

# Tokenize / Filter / Normalize
for index, row in data.iterrows():
    # Tokenization
    comment_tokens = re.split("[;., \-!?:\*]+", row['text'])
    
    #stemmer = arlstem.ARLSTem()
    #comment_tokens = list(map(stemmer.stem, comment_tokens))
        
    # Filtering
    filtered_words = [word for word in comment_tokens if word not in stopwords.words('arabic')]
    filtered_words = [re.sub('[^\u0621-\u0652]+', '', i) for i in filtered_words]
    filtered_words = list(filter(None, filtered_words))
    
    # Stemming
    #stemmer = arlstem.ARLSTem()
    #stemmed_words = list(map(stemmer.stem, filtered_words))

    # Normalization
    normalized = normalize.normalize_searchtext(' '.join(filtered_words))
    data.loc[index, 'text'] = normalized
    
data['text'].replace([''], np.nan, inplace=True)    
clean_data = data.dropna()
clean_data.to_excel('../feature datasets/ar/cleaned_data.xlsx', index_label="index")
