import os
from gensim.models import KeyedVectors
from flair.embeddings import WordEmbeddings
from gensim.models.keyedvectors import _try_upgrade
from tqdm import tqdm
import sys

def get_embedding(query):
    query = query.lower()
    embedding_path = os.path.join('w2v', "bible_word2vec_org")
    print('Loading pretrained bible embeddings from {} ...'.format(embedding_path))
    embeddings = KeyedVectors.load_word2vec_format(embedding_path)
    if query in embeddings.vocab.keys():
        ''' query word already in bible '''
        return embeddings[query]
    else:
        ''' query word not in bible '''
        fasttext_embeddings = WordEmbeddings('news')
        synonyms = fasttext_embeddings.precomputed_word_embeddings.most_similar(
            query, topn=500000)
        for synonym in tqdm(synonyms):
            if synonym in embeddings.vocab.keys():
                print('Using "{}" as synonym'.format(synonym))
                return embeddings[synonym]   
        
        raise Exception('No synonym in voab')

if __name__ == '__main__':
    get_embedding(sys.argv[1])
    # get_embedding(input())    