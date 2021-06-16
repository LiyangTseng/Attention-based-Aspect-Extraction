import os
from gensim.models import word2vec, KeyedVectors
import nltk
from nltk.corpus import gutenberg
from string import punctuation

def get_wv(bible_kjv_sents):
    home_path = os.path.expanduser('~')

    if not os.path.exists(os.path.join(home_path, 'nltk_data/corpora/gutenberg')):
        nltk.download('gutenberg')
    if not os.path.exists(os.path.join(home_path, 'nltk_data/tokenizers/punkt')):
        nltk.download('punkt')
    
    w2v_folder = 'w2v'
    if not os.path.exists(w2v_folder):
        os.mkdir(w2v_folder)
    embedding_path = os.path.join(w2v_folder, "bible_word2vec_org")
    vocab_path = os.path.join(w2v_folder, "bible_word2vec_vocalbulary")
    
    if not os.path.exists(embedding_path):
        print('generating word embeddings')
        # bible_kjv_sents = gutenberg.sents('bible-kjv.txt')
        # #  
        # discard_punctuation_and_lowercased_sents = [[word.lower() for word in sent if word not in punctuation] for sent in bible_kjv_sents]
        
        # bible_kjv_word2vec_model = word2vec.Word2Vec(discard_punctuation_and_lowercased_sents, min_count=1, size=200)
        bible_kjv_word2vec_model = word2vec.Word2Vec(bible_kjv_sents, min_count=1, size=200)
        bible_kjv_word2vec_model.wv.save_word2vec_format(embedding_path, vocab_path)
        print('bible word embeddings generated and stored at {}'.format(embedding_path))
        return  bible_kjv_word2vec_model.wv
    else:
        print('load word bible embeddings from {} '.format(embedding_path))
        return KeyedVectors.load_word2vec_format(embedding_path)

if __name__ == '__main__':
    print(get_wv().most_similar(['jesus']))