# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def tfidf_features(lang, input_data, m_features):
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
    cv.fit(data.text)
    cv_mat = cv.transform(data.text)
    
    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(cv_mat)
    
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': cv.get_feature_names(), 'weight': weights})
    weights_df.sort_values(by='weight', ascending=False).head(80)
    transformed_weights.toarray()
    
    tf_idf = pd.DataFrame(transformed_weights.todense(), index=data['index'], columns=cv.get_feature_names())
    
    
    tf_idf = tf_idf.add_prefix('tfidf:')
    return tf_idf