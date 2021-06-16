import os
from gensim.models import KeyedVectors

def get_embedding(query):
    embedding_path = os.path.join('w2v', "bible_word2vec_org")
    print('loading word bible embeddings from {} '.format(embedding_path))
    embeddings = KeyedVectors.load_word2vec_format(embedding_path)
    if query in embeddings.vocab.keys():
        ''' query word already in bible '''
        return embeddings[query]
    else:
        pass