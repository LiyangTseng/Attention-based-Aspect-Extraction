import os
from gensim.models import word2vec, KeyedVectors
import nltk
from nltk.corpus import gutenberg
from string import punctuation

def get_wv():
    home_path = os.path.expanduser('~')

    if not os.path.exists(os.path.join(home_path, 'nltk_data/corpora/gutenberg')):
        nltk.download('gutenberg')
    if not os.path.exists(os.path.join(home_path, 'nltk_data/tokenizers/punkt')):
        nltk.download('punkt')

    embedding_path = "bible_word2vec_org"
    if not os.path.exists(embedding_path):
        bible_kjv_sents = gutenberg.sents('bible-kjv.txt') 
        discard_punctuation_and_lowercased_sents = [[word.lower() for word in sent if word not in punctuation] for sent in bible_kjv_sents]
        bible_kjv_word2vec_model = word2vec.Word2Vec(discard_punctuation_and_lowercased_sents, min_count=1, size=200)
        bible_kjv_word2vec_model.wv.save_word2vec_format(embedding_path, "bible_word2vec_vocalbulary")
        print('bible word embedding generated and stored at {}'.format(embedding_path))
        return  bible_kjv_word2vec_model.wv
    else:
        print('load word bible embedding from {} '.format(embedding_path))
        return KeyedVectors.load_word2vec_format(embedding_path)

if __name__ == '__main__':
    print(get_wv().most_similar(['jesus']))