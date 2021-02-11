# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def count_features(lang, input_data, m_features):
    if lang == 'ar':
        if input_data == 'unclean':
            data = pd.read_excel('feature datasets/ar/labels.xlsx', encoding='utf-8')
        elif input_data == 'clean':
            data = pd.read_excel('feature datasets/ar/cleaned_data.xlsx', encoding='utf-8')
        elif input_data == 'clean_stemmed':
            data = pd.read_excel('feature datasets/ar/cleaned_data_stemmed.xlsx', encoding='utf-8')
    elif lang == 'en':
        if input_data == 'unclean':
            data = pd.read_excel('feature datasets/en/labels.xlsx', encoding='utf-8')
        elif input_data == 'clean':
            data = pd.read_excel('feature datasets/en/cleaned_data.xlsx', encoding='utf-8')
        elif input_data == 'clean_stemmed':
            data = pd.read_excel('feature datasets/en/cleaned_data_stemmed.xlsx', encoding='utf-8')
     
    cv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=m_features)
    transformed_counts = cv.fit_transform(data.text)
    counts = np.asarray(transformed_counts.mean(axis=0)).ravel().tolist()
    counts_df = pd.DataFrame({'term': cv.get_feature_names(), 'count': counts})
    counts_df.sort_values(by='count', ascending=False).head(80)
    transformed_counts.toarray()
    
    doc_freq = pd.DataFrame(transformed_counts.todense(), index=data['index'], columns=cv.get_feature_names())
    
    
    doc_freq = doc_freq.add_prefix('doc_freq:')
    return doc_freq