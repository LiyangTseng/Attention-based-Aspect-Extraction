import os
import numpy as np
from gensim.models import KeyedVectors
from scipy import spatial
import pandas as pd
from embedding import get_embedding
ASPECTS_NUM = 30

# input word
print('input your query keyword: ')
query = input()
    
query_embedding = get_embedding(query)
abae_centers_embeddings = np.load('abae_centers.npy')

# calculating cosine similarity with every aspects centers
max_similarity = -1
most_similar_aspect = 0
for aspect in range(ASPECTS_NUM):
    cosine_similarity = 1 - spatial.distance.cosine(query_embedding, abae_centers_embeddings[aspect, ])
    # argmax cosine similarity
    if cosine_similarity > max_similarity:
        max_similarity = cosine_similarity  
        most_similar_aspect = aspect

# aspect_probs
verse2aspect_path = 'verse2aspect.npy'
verse2aspect =  np.load(verse2aspect_path)

bible_verses_path = "./t_kjv.csv"
bible_df = pd.read_csv(bible_verses_path)

verses = bible_df['t'].values
aspects_qualified = (verse2aspect == most_similar_aspect)
verses = verses[aspects_qualified]
print(verses)