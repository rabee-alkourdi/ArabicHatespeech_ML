# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def word_bigrams_features(lang, input_data, m_features):
    if lang == 'ar':
        if input_data == 'unclean':
            data = pd.read_excel('feature datasets/ar/labels.xlsx', encoding='utf-8')
        elif input_data == 'clean':
            data = pd.read_excel('feature datasets/ar/cleaned_data.xlsx', encoding='utf-8')
        elif input_data == 'clean_stemmed':
            data = pd.read_excel('feature datasets/ar/cleaned_data_stemmed.xlsx', encoding='utf-8')
    elif lang == 'en':
        if input_data == 'unclean':
            data = pd.read_excel('feature datasets/en/labels.xlsx', encoding='ISO-8859-1')
        elif input_data == 'clean':
            data = pd.read_excel('feature datasets/en/cleaned_data.xlsx', encoding='ISO-8859-1')
        elif input_data == 'clean_stemmed':
            data = pd.read_excel('feature datasets/en/cleaned_data_stemmed.xlsx', encoding='ISO-8859-1')

        
    cv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,2), max_features=m_features)
    cv.fit(data.text)
    cv_mat = cv.transform(data.text)
    
    word_bigrams = pd.DataFrame(cv_mat.todense(), index=data['index'], columns=cv.get_feature_names())
    
    word_bigrams = word_bigrams.add_prefix('word_bigrams:')
    
    return word_bigrams
