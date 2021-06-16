import os
import numpy as np
from gensim.models import word2vec, KeyedVectors
from scipy import spatial
import pandas as pd

n_aspects = 30
bible_verses_path = "./t_kjv.csv"

# input word
print('input your keyword: ')
keyword = input()
    
# input to embedding
w2v_folder = 'w2v'
embedding_path = os.path.join(w2v_folder, "bible_word2vec_org")
print('load word bible embeddings from {} '.format(embedding_path))
embeddings = KeyedVectors.load_word2vec_format(embedding_path)
keyword_embedding = embeddings[keyword]
abae_centers = np.load('abae_centers.npy')

# calculating cosine similarity with every aspects centers
# argmax cosine similarity
max_similarity = -1
most_similar_aspect = 0
for aspect in range(n_aspects):
    cosine_similarity = 1 - spatial.distance.cosine(keyword_embedding, abae_centers[aspect, ])
    if cosine_similarity > max_similarity:
        max_similarity = cosine_similarity  
        most_similar_aspect = aspect

#  aspect_probs
verse2aspect_path = 'verse2aspect.npy'
verse2aspect =  np.load(verse2aspect_path)

aspects_qualified = (verse2aspect == most_similar_aspect)
bible_df = pd.read_csv(bible_verses_path)
verses = bible_df['t'].values
verses = verses[aspects_qualified]
print(verses)