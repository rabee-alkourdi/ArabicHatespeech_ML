# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import preprocessor as p
import string


data = pd.read_excel('../feature datasets/en/labels.xlsx',encoding = 'utf-8', index_col=0)

#p.set_options(p.OPT.HASHTAG)
stop_words = set(stopwords.words('english'))

clean_tweets = []
for index, row in data.iterrows():
    tweet = str(row['text']).lower()
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub("(rt @[A-Za-z0-9]+)",'',tweet)
    tweet = re.sub(r'\d+', '',tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = tweet.strip()
    
    tweet_tokens = word_tokenize(tweet)
    filtered_tweet = [word for word in tweet_tokens if word not in stop_words]
  
    clean_tweet = ' '.join(filtered_tweet)
    clean_tweets.append(clean_tweet)
    
data['text'] = clean_tweets

# Remove Empty Rows
data['text'].replace('', np.nan, inplace=True)
data.dropna(subset=['text'], inplace=True)

#stemmer = SnowballStemmer("english")
#data['text'] = data.text.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

'''
clean_tweets = []
for index, row in data.iterrows():
    tweet = str(row['text']).lower()
    
    initial_clean_tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)",'',tweet)
    clean_tweet_tokens = re.sub(r'\d+', '',initial_clean_tweet).split()
    clean_tweet = [word for word in clean_tweet_tokens if word not in stopwords.words('english')]
    clean_tweets.append(' '.join(clean_tweet_tokens))

data['text'] = clean_tweets

# Remove Duplicates
#data = data.drop_duplicates(subset='text', keep="first")

# Remove Empty Rows
data['text'].replace('', np.nan, inplace=True)
data.dropna(subset=['text'], inplace=True)

#stemmer = SnowballStemmer("english")
#data['text'] = data.text.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
'''

data.to_excel("../feature datasets/en/cleaned_data.xlsx", index_label="index")
