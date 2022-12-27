#!/usr/bin/env python
# coding: utf-8

# # HW 5: Clustering and Topic Modeling

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. </div>

# In this assignment, you'll practice different text clustering methods. A dataset has been prepared for you:
# - `hw5_train.csv`: This file contains a list of documents. It's used for training models
# - `hw5_test`: This file contains a list of documents and their ground-truth labels (4 lables: 1,2,3,7). It's used for external evaluation. 
# 
# |Text| Label|
# |----|-------|
# |paraglider collides with hot air balloon ... | 1|
# |faa issues fire warning for lithium ... | 2|
# | .... |...|
# 
# Sample outputs have been provided to you. Due to randomness, you may not get the exact result as shown here, but your result should be close if you tune the parameters carefully. Your taget is to `achieve above 70% F1 on the test dataset`

# ## Q1: K-Mean Clustering 
# 
# Define a function `cluster_kmean(train_text, test_text, text_label)` as follows:
# - Take three inputs: 
#     - `train_text` is a list of documents for traing 
#     - `test_text` is a list of documents for test
#     - `test_label` is the labels corresponding to documents in `test_text` 
# - First generate `TFIDF` weights. You need to decide appropriate values for parameters such as `stopwords` and `min_df`:
#     - Keep or remove stopwords? Customized stop words? 
#     - Set appropriate `min_df` to filter infrequent words
# - Use `KMeans` to cluster documents in `train_text` into 4 clusters. Here you need to decide the following parameters:
#     
#     - Distance measure: `cosine similarity`  or `Euclidean distance`? Pick the one which gives you better performance.  
#     - When clustering, be sure to  use sufficient iterations with different initial centroids to make sure clustering converge.
# - Test the clustering model performance using `test_label` as follows: 
#   - Predict the cluster ID for each document in `test_text`.
#   - Apply `majority vote` rule to dynamically map the predicted cluster IDs to `test_label`. Note, you'd better not hardcode the mapping, because cluster IDs may be assigned differently in each run. (hint: if you use pandas, look for `idxmax` function).
#   - Print out the cross tabluation between cluster ids and class labels
#   - print out the classification report for the test subset 
#   
#   
# - This function has no return. Print out the classification report. 
# 
# 
# - Briefly discuss:
#     - Which distance measure is better and why it is better. 
#     - Could you assign a meaningful name to each cluster? Discuss how you interpret each cluster.
# - Write your analysis in a pdf file.

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, mixture
from nltk.corpus import stopwords
from nltk.cluster import KMeansClusterer, cosine_distance
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation

# Add your import statement


# In[2]:


train = pd.read_csv("hw5_train.csv")
train.head()

test = pd.read_csv("hw5_test.csv")
test.head()

test_text = test["text"]
test_label = test["label"]
test_text


# In[18]:


def cluster_kmean(train, test_text, test_label):
    
   # Add your code
    tfidf_vect = TfidfVectorizer(stop_words = "english", min_df = 5)
    dtm = tfidf_vect.fit_transform(train)
    print(dtm.shape)
    
    num_clusters = 4
    
    clusterer = KMeansClusterer(num_clusters, cosine_distance, repeats = 20)
    clusters = clusterer.cluster(dtm.toarray(), assign_clusters = True)
    test_dtm = tfidf_vect.transform(test_text)
    predicted = [clusterer.classify(v) for v in test_dtm.toarray()]
    
    confusion_df = pd.DataFrame(list(zip(test_label, predicted)), columns = ["label", "cluster"])
    crosstab = pd.crosstab(index = confusion_df.cluster, columns = confusion_df.label)
    print('Crosstab of Kmean \n',crosstab)
    
    clusters_dict = crosstab.idxmax(axis = 'columns')
    predicted_targets = [clusters_dict[i] for i in predicted]
    
    print('\n Classification Report of Kmean\n',metrics.classification_report(test_label, predicted_targets))


# In[19]:


get_ipython().run_cell_magic('time', '', 'cluster_kmean(train["text"], test_text, test_label)')


# In[5]:


# cluster_kmean(train["text"], test_text, test_label)


# ## Q2: Clustering by Gaussian Mixture Model
# 
# In this task, you'll re-do the clustering using a Gaussian Mixture Model. Call this function  `cluster_gmm(train_text, test_text, text_label)`. 
# 
# Write your analysis on the following:
# - How did you pick the parameters such as the number of clusters, variance type etc.?
# - Compare to Kmeans in Q1, do you achieve better preformance by GMM? 
# 
# - Note, like KMean, be sure to use different initial means (i.e. `n_init` parameter) when fitting the model to achieve the model stability 

# In[24]:


def cluster_gmm(train, test_text, test_label):
    
    # Add your code
    tfidf_vect = TfidfVectorizer(stop_words = "english", min_df = 5)
    dtm = tfidf_vect.fit_transform(train)
    lowest_bic = np.infty
    best_gmm = None
    n_components_range = range(2,8)
    cv_types = ['spherical', 'tied', 'diag']
    
    for cv_type in cv_types:
        for n_component in n_components_range:
            gmm = mixture.GaussianMixture(n_components = n_component, covariance_type = cv_type, random_state = 42, n_init = 2)
            gmm.fit(dtm.toarray())
            bic = gmm.bic(dtm.toarray())
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
                
    test_dtm = tfidf_vect.transform(test_text)
    predicted = best_gmm.predict(test_dtm.toarray())
    
    confusion_df = pd.DataFrame(list(zip(test_label, predicted)), columns = ["label", "cluster"])
    crosstab = pd.crosstab(index = confusion_df.cluster, columns = confusion_df.label)
    print('Crosstab \n',crosstab)
    
    clusters_dict = crosstab.idxmax(axis = 'columns')
    predicted_targets = [clusters_dict[i] for i in predicted]
    
    print('\nClassification Report for GMM\n',metrics.classification_report(test_label, predicted_targets))


# In[25]:


get_ipython().run_cell_magic('time', '', 'cluster_gmm(train["text"], test_text, test_label)')


# ## Q3: Clustering by LDA 
# 
# In this task, you'll re-do the clustering using LDA. Call this function `cluster_lda(train_text, test_text, text_label)`. 
# 
# However, since LDA returns topic mixture for each document, you `assign the topic with highest probability to each test document`, and then measure the performance as in Q1
# 
# In addition, within the function, please print out the top 30 words for each topic
# 
# Finally, please analyze the following:
# - Based on the top words of each topic, could you assign a meaningful name to each topic?
# - Although the test subset shows there are 4 clusters, without this information, how do you choose the number of topics? 
# - Does your LDA model achieve better performance than KMeans or GMM?

# In[22]:


def cluster_lda(train, test_text, test_label):
    
    # add your code
    tf_vectorizer = CountVectorizer(min_df = 5, stop_words=list(stopwords.words('english')))
    tf = tf_vectorizer.fit_transform(train.tolist()+test_text.tolist())
    tf_feature_names = tf_vectorizer.get_feature_names()
    num_topics = 4
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=30, verbose=1, evaluate_every=1, n_jobs=1, random_state=0).fit(tf)
    num_top_words = 30
    
    for topic_idx, topic in enumerate(lda.components_):
        print ("Topic %d:" % (topic_idx))
        words=[tf_feature_names[i] for i in topic.argsort()[::-1][0:num_top_words]]
        print(words)
        print("\n")
        
    topic_assign = lda.transform(tf)
    predicted = []
    
    for i in range(4000, 5274):
        x = np.array(topic_assign[i])
        predicted.append(np.argmax(x))
        
    confusion_df = pd.DataFrame(list(zip(test_label, predicted)), columns = ["label", "cluster"])
    crosstab = pd.crosstab(index = confusion_df.cluster, columns = confusion_df.label)
    print('Crosstab for LDA \n',crosstab)
    
    clusters_dict = crosstab.idxmax(axis = 'columns')
    predicted_targets = [clusters_dict[i] for i in predicted]
    
    print('\n Classification for LDA \n',metrics.classification_report(test_label, predicted_targets))


# In[23]:


get_ipython().run_cell_magic('time', '', 'cluster_lda(train["text"], test_text, test_label)')


# In[10]:


# cluster_lda(train["text"], test_text, test_label)


# ## Q4 (Bonus): Topic Coherence and Separation
# 
# For the LDA model you obtained at Q3, can you measure the coherence and separation of topics? Suppose you have the following topics:
# - Topic 1 keywords: business, money, company, pay, credit
# - Topic 2 keywords: energy, earth, gas, heat, sun
# 
# Describe your ideas and implement them.
# 
# 

# In[26]:


if __name__ == "__main__":  
    
    # Due to randomness, you won't get the exact result
    # as shown here, but your result should be close
    # if you tune the parameters carefully
    
    train = pd.read_csv("hw5_train.csv")
    train.head()

    test = pd.read_csv("hw5_test.csv")
    test.head()

    test_text = test["text"]
    test_label = test["label"]
    
    # Q1
    cluster_kmean(train["text"], test_text, test_label)
            
    # Q2
    cluster_gmm(train["text"], test_text, test_label)
    
    # Q2
    cluster_lda(train["text"], test_text, test_label)


# In[ ]:




