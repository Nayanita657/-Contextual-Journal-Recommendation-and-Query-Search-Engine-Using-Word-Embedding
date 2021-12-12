#!/usr/bin/env python
# coding: utf-8

# ## Embeddings by SentenceBERT

# In[55]:


# https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/
# glove https://nlp.stanford.edu/pubs/glove.pdf
import pandas as pd
import numpy as np


# In[71]:


from sentence_transformers import SentenceTransformer
class Bert_embedding:
    
    def __init__(self, df_sentances):
        self.sentances = list(df_sentances['sentances']) # cleaned and corrected sentances
        self.query = None # cleaned and corrected query
        self.df_sentances = df_sentances
        self.sentence_embeddings = None
        self.sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        
        
    def cosine_similarity(self, a, b):
        nominator = np.dot(a, b)

        a_norm = np.sqrt(np.sum(a**2))
        b_norm = np.sqrt(np.sum(b**2))

        denominator = a_norm * b_norm

        cosine_similarity = nominator / denominator

        return cosine_similarity
    
    def query_embedding(self, query, sbert_model):
        query_vector= sbert_model.encode(query)

        return query_vector

    
    
    def fit(self):
        self.sentence_embeddings = self.sbert_model.encode(self.sentances)
    
    
    def fit_transform(self, query):
        self.query = query
        query_vector = self.query_embedding(self.query, self.sbert_model)

        cosine_score = []
        for i in self.sentence_embeddings:
            cosine_score.append(self.cosine_similarity(query_vector,i))


        self.df_sentances['cosine_score'] = cosine_score

        self.df_sentances = self.df_sentances.sort_values(by = 'cosine_score',ascending = False)
        self.df_sentances.reset_index(inplace=True,drop=True)
    
        return self.df_sentances


# In[72]:


class glove50_embedding:
    
    def __init__(self, df_sentances):
        self.sentances = list(df_sentances['sentances']) # cleaned and corrected sentances
        self.query = None # cleaned and corrected query
        self.df_sentances = df_sentances
        self.sentence_embeddings = None
        self.word_embeddings = None
        
    def glove_vector(self):
        self.word_embeddings = {}

        f = open('/home/swapnil/Documents/ML1/Project/Implementation/glove.6B/glove.6B.50d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            self.word_embeddings[word] = coefs
        f.close()
        
    def cosine_similarity(self, a, b):
        nominator = np.dot(a, b)

        a_norm = np.sqrt(np.sum(a**2))
        b_norm = np.sqrt(np.sum(b**2))

        denominator = a_norm * b_norm

        cosine_similarity = nominator / denominator

        return cosine_similarity
    
    def query_embedding(self, query):
        query_vector= sum([self.word_embeddings.get(w, np.zeros((50,))) for w in query.split()])/(len(query.split())+0.001)

        return query_vector

    
    
    def fit(self):
        
        self.glove_vector()
        
        self.sentence_embeddings = []
        cnt = 0
        for i in self.sentances:
            if len(i) != 0:
                v = sum([self.word_embeddings.get(w, np.zeros((50,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((50,))
                cnt=cnt+1
            self.sentence_embeddings.append(v)
    
    
    def fit_transform(self, query):
        self.query = query
        query_vector = self.query_embedding(self.query)

        cosine_score = []
        for i in self.sentence_embeddings:
            cosine_score.append(self.cosine_similarity(query_vector,i))


        self.df_sentances['cosine_score'] = cosine_score

        self.df_sentances = self.df_sentances.sort_values(by = 'cosine_score',ascending = False)
        self.df_sentances.reset_index(inplace=True,drop=True)
    
        return self.df_sentances


# In[79]:


# !pip install "tensorflow>=2.0.0"
# !pip install --upgrade tensorflow-hub

# import necessary libraries
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class elmo_embedding:
    
    def __init__(self, df_sentances):
        self.sentances = list(df_sentances['sentances']) # cleaned and corrected sentances
        self.query = None # cleaned and corrected query
        self.df_sentances = df_sentances
        self.sentence_embeddings = None
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        
        
        
    def elmo_vectors(self, x):
        embeddings = self.elmo(x, signature="default", as_dict=True)["elmo"]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings,1))
        
    def cosine_similarity(self, a, b):
        nominator = np.dot(a, b)

        a_norm = np.sqrt(np.sum(a**2))
        b_norm = np.sqrt(np.sum(b**2))

        denominator = a_norm * b_norm

        cosine_similarity = nominator / denominator

        return cosine_similarity
    
    def query_embedding(self, query):
        query_vector= self.elmo_vectors([query])

        return query_vector

    
    
    def fit(self):
        # Extract ELMo embeddings
        self.sentence_embeddings = self.elmo_vectors(self.sentances)
        
    
    
    def fit_transform(self, query):
        self.query = query
        query_vector = self.query_embedding(self.query)

        cosine_score = []
        for i in self.sentence_embeddings:
            cosine_score.append(self.cosine_similarity(query_vector,i))


        self.df_sentances['cosine_score'] = cosine_score

        self.df_sentances = self.df_sentances.sort_values(by = 'cosine_score',ascending = False)
        self.df_sentances.reset_index(inplace=True,drop=True)
    
        return self.df_sentances


# In[ ]:




