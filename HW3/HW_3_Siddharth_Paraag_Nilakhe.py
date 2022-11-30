#!/usr/bin/env python
# coding: utf-8

# # HW 4: Natural Language Processing

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work or let someone copy your solution (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. No last minute extension of due date. Be sure to start working on it ASAP! </div>

# ## Q1: Extract data using regular expression
# Suppose you have scraped the text shown below from an online source (https://www.google.com/finance/). 
# Define a `extract` function which:
# - takes a piece of text (in the format of shown below) as an input
# - uses regular expression to transform the text into a DataFrame with columns: 'Ticker','Name','Article','Media','Time','Price',and 'Change' 
# - returns the DataFrame

# In[60]:


import pandas as pd
import nltk
from sklearn.metrics import pairwise_distances
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import re
import spacy

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[61]:


text = '''QQQ
Invesco QQQ Trust Series 1
Invesco Expands QQQ Innovation Suite to Include Small-Cap ETF
PR Newswire • 4 hours ago
$265.62
1.13%
add_circle_outline
AAPL
Apple Inc
Estimating The Fair Value Of Apple Inc. (NASDAQ:AAPL)
Yahoo Finance • 4 hours ago
$140.41
1.50%
add_circle_outline
TSLA
Tesla Inc
Could This Tesla Stock Unbalanced Iron Condor Return 23%?
Investor's Business Daily • 1 hour ago
$218.30
0.49%
add_circle_outline
AMZN
Amazon.com, Inc.
The Regulators of Facebook, Google and Amazon Also Invest in the Companies' Stocks
Wall Street Journal • 2 days ago
$110.91
1.76%
add_circle_outline'''


# In[62]:


def extract(text):
    
    result = None
    # add your code here
    
    data_re = re.findall('([A-Z]+)\n([A-Za-z\s\.,\d]+)\n([A-Za-z-%\.\d\?,\)\(\':\s]+)\n([A-Za-z\'\s]+)•\s([A-Za-z\d\s]+)\n([\d\$\.]+)\n([\d\.%]+)' , text)
    
    result = pd.DataFrame(data_re,columns=['Ticker','Name','Article','Media','Time','Price','Change'])
    
    return result


# In[63]:


# test your function

extract(text)


# ## Q2: Analyze a document
# 
# When you have a long document, you would like to 
# - Quanitfy how `concrete` a sentence is
# - Create a concise summary while preserving it's key information content and overall meaning. Let's implement an `extractive method` based on the concept of TF-IDF. The idea is to identify the key sentences from an article and use them as a summary. 
# 
# 
# Carefully follow the following steps to achieve these two targets.

# ### Q2.1. Preprocess the input document 
# 
# Define a function `proprocess(doc, lemmatized = True, remove_stopword = True, lower_case = True, remove_punctuation = True, pos_tag = False)` 
# - Four input parameters:
#     - `doc`: an input string (e.g. a document)
#     - `lemmatized`: an optional boolean parameter to indicate if tokens are lemmatized. The default value is True (i.e. tokens are lemmatized).
#     - `remove_stopword`: an optional boolean parameter to remove stop words. The default value is True, i.e., remove stop words. 
#     - `remove_punctuation`: optional boolean parameter to remove punctuations. The default values is True, i.e., remove all punctuations.
#     - `lower_case`: optional boolean parameter to convert all tokens to lower case. The default option is True, i.e., lowercase all tokens.
#     - `pos_tag`: optional boolean parameter to add a POS tag for each token. The default option is False, i.e., no POS tagging.  
#     
#        
# - Split the input `doc` into sentences. Hint, typically, `\n\n+` is used to separate paragraphs. Make sure a sentence does not cross over two paragraphs. You can replace `\n\n+` by a `.`
# 
# 
# - Tokenize each sentence into unigram tokens and also process the tokens as follows:
#     - If `lemmatized` is True, lemmatize all unigrams. 
#     - If `remove_stopword` is set to True, remove all stop words. 
#     - If `remove_punctuation` is set to True, remove all punctuations. 
#     - If `lower_case` is set to True, convert all tokens to lower case 
#     - If `pos_tag` is set to True, find the POS tag for each token and form a tuple for each token, e.g., ('recently', 'ADV'). Either Penn tags or Universal tags are fine. See mapping of these two tagging systems here: https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
# 
# 
# - Return the original sentence list (`sents`) and also the tokenized (or tagged) sentence list (`tokenized_sents`). 
# 
#    
# (Hint: you can use [nltk](https://www.nltk.org/api/nltk.html) and [spacy](https://spacy.io/api/token#attributes) package for this task.)

# In[64]:


nlp = spacy.load("en_core_web_sm")
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
def preprocess(doc, lemmatized=True, pos_tag = False, remove_stopword=True, lower_case = True, remove_punctuation = True):
   
    sents, tokenized_sents = None, None
    
    # add your code here
    tokenized_sents = []
    doc = doc.replace('\n\n','. ')
    
    sents = sent_tokenize(doc)
    stop_words = set(stopwords.words("english"))
    s=set(string.punctuation)
    for s in sents:
        
        temp = word_tokenize(s)
        
        if lower_case == True:
            temp = [x.lower() for x in temp]
            
        if remove_punctuation == True:
            temp = [x for x in temp if x not in string.punctuation]
        
        if remove_stopword == True:
            temp = [x for x in temp if x not in stop_words]
        
        check_pos = [(x, nlp(x)[0].pos_) for x in temp]
        pos_lemma = []
        for x in enumerate(check_pos):
            if x[1][1] in ['ADJ']:
                pos_lemma.append((x[1][0], wordnet.ADJ))
            elif x[1][1] in ['VERB']:
                pos_lemma.append((x[1][0], wordnet.VERB))
            elif x[1][1] in ['ADV']:
                pos_lemma.append((x[1][0], wordnet.ADV))
            else:
                pos_lemma.append((x[1][0], wordnet.NOUN))
                
        if lemmatized == True:
            temp = [lemmatizer.lemmatize(x[0], x[1]) for x in pos_lemma]
  
        if pos_tag == True:
            temp = [(x, nlp(x)[0].pos_) for x in temp]
    
        tokenized_sents.append(temp)
    
    return sents, tokenized_sents


# In[65]:


# load test document

text = open("power_of_nlp.txt", "r", encoding='utf-8').read()


# In[66]:


# test with all default options:

sents, tokenized_sents = preprocess(text)

# print first 3 sentences
for i in range(3):
    print(sents[i], "\n",tokenized_sents[i],"\n\n" )


# In[67]:


# process text without remove stopwords, punctuation, lowercase, but with pos tagging

sents, tokenized_sents = preprocess(text, lemmatized = False, pos_tag = True, 
                                    remove_stopword=False, remove_punctuation = False, 
                                    lower_case = False)

for i in range(3):
    print(sents[i], "\n",tokenized_sents[i],"\n\n" )


# ### Q2.2. Quantify sentence concreteness
# 
# 
# `Concreteness` can increase a message's persuasion. The concreteness can be measured by the use of :
# - `article` (e.g., a, an, and the), 
# - `adpositions` (e.g., in, at, of, on, etc), and
# - `quantifiers`, i.e., adjectives before nouns.
# 
# 
# Define a function `compute_concreteness(tagged_sent)` as follows:
# - Input argument is `tagged_sent`, a list with (token, pos_tag) tuples as shown above.
# - Find the three types of tokens: `articles`, `adposition`, and `quantifiers`.
# - Compute `concereness` score as:  `(the sum of the counts of the three types of tokens)/(total non-punctuation tokens)`.
# - return the concreteness score, articles, adposition, and quantifiers lists.
# 
# 
# Find the most concrete and the least concrete sentences from the article. 
# 
# 
# Reference: Peer to Peer Lending: The Relationship Between Language Features, Trustworthiness, and Persuasion Success, https://socialmedialab.sites.stanford.edu/sites/g/files/sbiybj22976/files/media/file/larrimore-jacr-peer-to-peer.pdf

# In[68]:


def compute_concreteness(tagged_sent):
    
    concreteness, articles, adpositions,quantifier = None, None, None, []
    
    # add your code here
    articles=[ x for x in tagged_sent          if x[0].lower() == 'a' or x[0].lower() == 'an' or x[0].lower() == 'the']
    
    adpositions=[ x for x in tagged_sent          if x[1].startswith('ADP') ]
    
    for x in range(len(tagged_sent)):
        if x<len(tagged_sent) and tagged_sent[x][1] == 'ADJ' and tagged_sent[x+1][1] == 'NOUN':
            quantifier.append(tagged_sent[x])
            
    non_puncts = [x for x in tagged_sent                  if x[1]!='PUNCT']
    number_non_puncts = len(non_puncts)
    
    concreteness = (len(adpositions) + len(articles) + len(quantifier)) / number_non_puncts
    
    return concreteness, articles, adpositions,quantifier


# In[69]:


# tokenize with pos tag, without change the text much

sents, tokenized_sents = preprocess(text, lemmatized = False, pos_tag = True, 
                                    remove_stopword=False, remove_punctuation = False, 
                                    lower_case = False)


# In[70]:


# find concreteness score, articles, adpositions, and quantifiers in a sentence

idx = 1    # sentence id
x = tokenized_sents[idx]
concreteness, articles, adpositions,quantifier = compute_concreteness(x)

# show sentence
sents[idx]
# show result
concreteness, articles, adpositions,quantifier


# In[71]:


# Find the most concrete and the least concrete sentences from the article
concrete = []
# add your code here

for x in tokenized_sents:
    concreteness, articles, adpositions,quantifier = compute_concreteness(x)
    concrete.append(concreteness)
max_id = concrete.index(max(concrete))
min_id = concrete.index(min(concrete))


print (f"The most concerete sentence:  {sents[max_id]}, {concrete[max_id]:.3f}\n")
print (f"The least concerete sentence:  {sents[min_id]}, {concrete[min_id]:.3f}")


# ### Q2.3. Generate TF-IDF representations for sentences 
# 
# Define a function `compute_tf_idf(sents, use_idf)` as follows: 
# 
# 
# - Take the following two inputs:
#     - `sents`: tokenized sentences (without pos tagging) returned from Q2.1. These sentences form a corpus for you to calculate `TF-IDF` vectors.
#     - `use_idf`: if this option is true, return smoothed normalized `TF_IDF` vectors for all sentences; otherwise, just return normalized `TF` vector for each sentence.
#     
#     
# - Calculate `TF-IDF` vectors as shown in the lecture notes (Hint: you can slightly modify code segment 7.5 in NLP Lecture Notes (II) for this task)
# 
# - Return the `TF-IDF` vectors  if `use_idf` is True.  Return the `TF` vectors if `use_idf` is False.

# In[72]:



def compute_tf_idf(sents, use_idf = True, min_df = 1):
    
    tf_idf = None
    # add your code here
    docs_tokens = {idx: (nltk.FreqDist(sent))
                 for idx,sent in enumerate(sents)}
    dtm = pd.DataFrame.from_dict(docs_tokens,                            orient="index" )
    dtm = dtm.fillna(0)
    dtm = dtm.sort_index(axis = 0)
    
    tf = dtm.values
    doc_len = tf.sum(axis = 1, keepdims = True)
    tf = np.divide(tf, doc_len)
    
    df = np.where(tf > 0, 1, 0)

    smoothed_idf = np.log(np.divide(len(sents) + 1, np.sum(df, axis=0) + 1)) + 1    
    smoothed_tf_idf = tf * smoothed_idf
          
    if use_idf:
        return smoothed_tf_idf
    else:
        return tf


# In[73]:


# test compute_tf_idf function

sents, tokenized_sents = preprocess(text)
tf_idf= compute_tf_idf(tokenized_sents, use_idf = True)

# show shape of TF-IDF
tf_idf.shape


# ### Q2.4. Identify key sentences as summary 
# 
# The basic idea is that, in a coherence article, all sentences should center around some key ideas. If we can identify a subset of sentences, denoted as $S_{key}$, which precisely capture the key ideas,  then $S_{key}$ can be used as a summary. Moreover, $S_{key}$ should have high similarity to all the other sentences on average, because all sentences are centered around the key ideas contained in $S_{key}$. Therefore, we can identify whether a sentence belongs to $S_{key}$ by its similarity to all the other sentences.
# 
# 
# Define a function `get_summary(tf_idf, sents, topN = 5)`  as follows:
# 
# - This function takes three inputs:
#     - `tf_idf`: the TF-IDF vectors of all the sentences in a document
#     - `sents`: the original sentences corresponding to the TF-IDF vectors
#     - `topN`: the top N sentences in the generated summary
# 
# - Steps:
#     1. Calculate the cosine similarity for every pair of TF-IDF vectors 
#     1. For each sentence, calculate its average similarity to all the others 
#     1. Select the sentences with the `topN` largest average similarity 
#     1. Print the `topN` sentences index
#     1. Return these sentences as the summary

# In[74]:


def get_summary(tf_idf, sents, topN = 5):
    
    summary = []
    
    # add your code here
    similarity=1-pairwise_distances(tf_idf, metric = 'cosine')
    check = []
    for item in similarity:
        x = sum(item)/len(item)
        check.append(x)
    top_idx = np.argsort(check)[-topN:]
    top_idx= np.flip(top_idx)
    top_values = [check[i] for i in top_idx]
    
    for i in top_idx:
        summary.append(sents[i])

    
    return summary


# In[75]:


# put everything together and test with different options

sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
summary = get_summary(tf_idf, sents, topN = 5)

for sent in summary:
    print(sent,"\n")


# ## Trying out different metrics for the get_summary function

# ### Metric - cityblock

# In[76]:


# Please test summary generated under different configurations

def get_summary_1(tf_idf, sents, topN = 5):
    
    summary = []
    
    # add your code here
    similarity=1-pairwise_distances(tf_idf, metric = 'cityblock')
    check = []
    for item in similarity:
        x = sum(item)/len(item)
        check.append(x)
    top_idx = np.argsort(check)[-topN:]
    top_idx= np.flip(top_idx)
    top_values = [check[i] for i in top_idx]
    
    for i in top_idx:
        summary.append(sents[i])

    
    return summary


# In[77]:



sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
summary = get_summary_1(tf_idf, sents, topN = 5)

for sent in summary:
    print(sent,"\n")


# ### Metric - euclidean

# In[78]:


def get_summary_2(tf_idf, sents, topN = 5):
    
    summary = []
    
    # add your code here
    similarity=1-pairwise_distances(tf_idf, metric = 'euclidean')
    check = []
    for item in similarity:
        x = sum(item)/len(item)
        check.append(x)
    top_idx = np.argsort(check)[-topN:]
    top_idx= np.flip(top_idx)
    top_values = [check[i] for i in top_idx]
    
    for i in top_idx:
        summary.append(sents[i])

    
    return summary


# In[79]:



sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
summary = get_summary_2(tf_idf, sents, topN = 5)

for sent in summary:
    print(sent,"\n")


# ### Metric - L1

# In[80]:


def get_summary_3(tf_idf, sents, topN = 5):
    
    summary = []
    
    # add your code here
    similarity=1-pairwise_distances(tf_idf, metric = 'l1')
    check = []
    for item in similarity:
        x = sum(item)/len(item)
        check.append(x)
    top_idx = np.argsort(check)[-topN:]
    top_idx= np.flip(top_idx)
    top_values = [check[i] for i in top_idx]
    
    for i in top_idx:
        summary.append(sents[i])

    
    return summary


# In[81]:



sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
summary = get_summary_3(tf_idf, sents, topN = 5)

for sent in summary:
    print(sent,"\n")


# ### Metric - L2

# In[82]:


def get_summary_4(tf_idf, sents, topN = 5):
    
    summary = []
    
    # add your code here
    similarity=1-pairwise_distances(tf_idf, metric = 'l2')
    check = []
    for item in similarity:
        x = sum(item)/len(item)
        check.append(x)
    top_idx = np.argsort(check)[-topN:]
    top_idx= np.flip(top_idx)
    top_values = [check[i] for i in top_idx]
    
    for i in top_idx:
        summary.append(sents[i])

    
    return summary


# In[83]:



sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
summary = get_summary_4(tf_idf, sents, topN = 5)

for sent in summary:
    print(sent,"\n")


# ###  Metric - Manhattan

# In[84]:


def get_summary_5(tf_idf, sents, topN = 5):
    
    summary = []
    
    # add your code here
    similarity=1-pairwise_distances(tf_idf, metric = 'manhattan')
    check = []
    for item in similarity:
        x = sum(item)/len(item)
        check.append(x)
    top_idx = np.argsort(check)[-topN:]
    top_idx= np.flip(top_idx)
    top_values = [check[i] for i in top_idx]
    
    for i in top_idx:
        summary.append(sents[i])

    
    return summary


# In[85]:



sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
summary = get_summary_5(tf_idf, sents, topN = 5)

for sent in summary:
    print(sent,"\n")


# ### Q2.5. Analysis 
# 
# - Do you think the way to quantify concreteness makes sense? Any other thoughts to measure concreteness or abstractness? Share your ideas in pdf.
# 
# 
# - Do you think this method is able to generate a good summary? Any pros or cons have you observed? 
# 
# 
# - Do these options `lemmatized, remove_stopword, remove_punctuation, use_idf` matter? 
# - Why do you think these options matter or do not matter? 
# - If these options matter, what are the best values for these options?
# 
# 
# Write your analysis as a pdf file. Be sure to provide some evidence from the output of each step to support your arguments.

# ### Q2.5. (Bonus 3 points). 
# 
# 
# - Can you think a way to improve this extractive summary method? Explain the method you propose for improvement,  implement it, use it to generate a new summary, and demonstrate what is improved in the new summary.
# 
# 
# - Or, you can research on some other extractive summary methods and implement one here. Compare it with the one you implemented in Q2.1-Q2.3 and show pros and cons of each method.

# In[86]:


def get_summary_6(tf_idf, sents, topN = 5):
    
    summary = []
    
    # add your code here
    similarity=1-pairwise_distances(tf_idf, metric = 'jaccard')
    check = []
    for item in similarity:
        x = sum(item)/len(item)
        check.append(x)
    top_idx = np.argsort(check)[-topN:]
    top_idx= np.flip(top_idx)
    top_values = [check[i] for i in top_idx]
    
    for i in top_idx:
        summary.append(sents[i])

    
    return summary


# In[87]:



sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
summary = get_summary_6(tf_idf, sents, topN = 5)

for sent in summary:
    print(sent,"\n")


# ## Main block to test all functions

# In[88]:


if __name__ == "__main__":  
    
    
    text=text = '''QQQ
Invesco QQQ Trust Series 1
Invesco Expands QQQ Innovation Suite to Include Small-Cap ETF
PR Newswire • 4 hours ago
$265.62
1.13%
add_circle_outline
AAPL
Apple Inc
Estimating The Fair Value Of Apple Inc. (NASDAQ:AAPL)
Yahoo Finance • 4 hours ago
$140.41
1.50%
add_circle_outline
TSLA
Tesla Inc
Could This Tesla Stock Unbalanced Iron Condor Return 23%?
Investor's Business Daily • 1 hour ago
$218.30
0.49%
add_circle_outline
AMZN
Amazon.com, Inc.
The Regulators of Facebook, Google and Amazon Also Invest in the Companies' Stocks
Wall Street Journal • 2 days ago
$110.91
1.76%
add_circle_outline'''
    
    
    print("\n==================\n")
    print("Test Q1")
    print(extract(text))
    
    print("\n==================\n")
    print("Test Q2.1")
    
    text = open("power_of_nlp.txt", "r", encoding='utf-8').read()
    
    sents, tokenized_sents = preprocess(text, lemmatized = False, pos_tag = True, 
                                    remove_stopword=False, remove_punctuation = False, 
                                    lower_case = False)
    
    idx = 1    # sentence id
    x = tokenized_sents[idx]
    concreteness, articles, adpositions,quantifier = compute_concreteness(x)

    # show sentence
    sents[idx]
    # show result
    concreteness, articles, adpositions,quantifier
    
    print("\n==================\n")
    print("Test Q2.2-2.4")
    sents, tokenized_sents = preprocess(text)
    tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
    summary = get_summary(tf_idf, sents, topN = 5)
    print(summary)


# In[ ]:




