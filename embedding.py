import os
from gensim.models import KeyedVectors
from flair.embeddings import WordEmbeddings
import numpy as np
from tqdm import tqdm
import sys

def avg_embedding(l):
    ''' averege on different word embeddings '''
    z = list(zip(*l))
    sum_of_lists = list(map(sum, z))
    return np.array(sum_of_lists)

def get_embedding(query):
    query = query.lower()
    embedding_path = os.path.join('w2v', "bible_word2vec_org")
    print('Checking if query in bible ...')
    embeddings = KeyedVectors.load_word2vec_format(embedding_path)
    if query in embeddings.vocab.keys():
        ''' query word already in bible '''
        print('Query exists in Bible')
        return embeddings[query]
    else:
        ''' query word not in bible '''
        print('Query not found in Bible')
        fasttext_embeddings = WordEmbeddings('news')
        synonyms = fasttext_embeddings.precomputed_word_embeddings.most_similar(
            query, topn=50000)
        synonym_found = False
        # TODO change to avg embedding
        print_cnt=0
        possible_synonyms = []
        for synonym_tuple in tqdm(synonyms, desc='Searching Synonyms'):
            if synonym_tuple[0] in embeddings.vocab.keys():
                if print_cnt<5:
                    print(f'synonym={synonym_tuple[0]}')
                    print(f'similarity={synonym_tuple[1]}')
                    print('-'*50)
                    print_cnt+=1
                possible_synonyms.append(embeddings[synonym_tuple[0]]*synonym_tuple[1])
        
        synonym_found = len(possible_synonyms)>0
        if synonym_found:
            print('Using "{}" as synonym'.format(synonym_tuple[0]))
            return avg_embedding(possible_synonyms)  
        else:
            raise Exception('No synonyms found in Bible')

        # for synonym_tuple in tqdm(synonyms, desc='Searching Synonyms'):
        #     if synonym_tuple[0] in embeddings.vocab.keys():
        #         synonym_found = True
        #         break
        
        # if synonym_found:
        #     print('Using "{}" as synonym'.format(synonym_tuple[0]))
        #     return embeddings[synonym_tuple[0]]   
        # else:
        #     raise Exception('No synonyms found in Bible')

if __name__ == '__main__':
    get_embedding(sys.argv[1])
    # get_embedding(input())    