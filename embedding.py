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
        print('1-------------')
        model = KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False)
        word_vectors = model.wv
        # -- this saves space, if you plan to use only, but not to train, the model:
        del model

        # -- do your work:
        similar_words = word_vectors.most_similar(query, topn=10) 
        print(similar_words)
if __name__ == '__main__':
    get_embedding('technology')
        