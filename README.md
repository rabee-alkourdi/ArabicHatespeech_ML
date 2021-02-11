# Detecting and Modeling Anti-Social Behavior in Social Networks

Abstract

There are a few studies regarding the problem of automatically detecting antisocial behavior such as hate speech, bullying, harassment, trolling, and rumor in social networks on Arabic content in contrast to those done on English content, despite the danger of said behavior and its huge negative impact on both communities and individuals alike. 
This study tackles the problem by training two predictive models to detect this behavior. The first is built using the famous Support Vector Machine algorithm, while the second is built using Convolutional Neural Networks.
 This study tests the performance of both models in classifying antisocial behavior in both Arabic and English content. It also compares their performance to determine the impact a language has on the model’s performance. Based on that, an architecture was suggested for the neural network which aims to get a good classifier for Arabic data and then compare it to another trained on English data. This research also studies the effect of data preprocessing techniques on the performance of the model. It also tests the effect of several features on the classification which are statistical features such as tf-idf, grammatical features such as Parts of Speech and Dependency Relations, and contextual features such as Word Embeddings. Finally, the model with the best performance in classifying hate speech within Arabic and English content is selected after being evaluated using some metrics such as precision, recall, and accuracy.
The results of this study show the superiority of the CNN model compared to its SVM counterpart for both datasets. It had an approximate detection rate of 79% for hate speech in Arabic comments and 98% in English tweets, which also shows the superiority of the English model to the Arabic one. Therefore, there is much room for improvement. The study also proves the positive impact of data preprocessing on the model’s performance for what it does in regards of noise removal in data.

This code includes:
- Data preprocessing: Tokenization, Filtering, Normalization, Stemming
- Feature Representation: tf-idf, PoS, Dependency Relations, Word Embeddings (fastText)
- Building a ML model using SVM
- Building a ML model using CNN
