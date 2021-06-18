import os
import numpy as np
from scipy import spatial
import pandas as pd
from embedding import get_embedding
import argparse
from gensim.models import KeyedVectors

ASPECTS_NUM = 30
BOOK_NAME = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua', 'Judges', 'Ruth', '1 Samuel',
            '2 Samuel', '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 
            'Job', 'Psalm', 'Proverbs', 'Ecclesiastes', 'Song of Solomon', 'Isaiah', 'Jeremiah', 
            'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah',
            'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi', 'Matthew', 'Mark',
            'Luke', 'John', 'Acts', 'Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians',
            'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians', '1 Timothy', '2 Timothy',
            'Titus', 'Philemon', 'Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John',
            'Jude', 'Revelation']

def find_verse():
    '''
    python test.py [query] -a [True/False] -s [True/False] -w [True/False], see details in embedding.py
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('query', help='query keyword')
    parser.add_argument('-a', '--use_ABAE', help='use ABAE or not', action='store_false')
    parser.add_argument('-s', '--sort_by_rev', help='sort by revelence or not', action='store_false')
    parser.add_argument('-w', '--use_weighted_embedding', help='use weighted embeddings or not', action='store_false')
    args = parser.parse_args()
    
    query = args.query
    use_ABAE = args.use_ABAE
    sort_by_relevence = args.sort_by_rev
    use_weighted_embedding = args.use_weighted_embedding
    print('Using ABAE: ', use_ABAE)
    
    bible_verses_path = "./t_kjv.csv"
    bible_df = pd.read_csv(bible_verses_path)

    w2v_folder = 'w2v'
    embedding_path = os.path.join(w2v_folder, "bible_verse_org")
    verse_vector = KeyedVectors.load_word2vec_format(embedding_path)

    output_folder = 'output'
    att_embedding_path = os.path.join(w2v_folder, "bible_verse_att_org")
    att_verse_vector = KeyedVectors.load_word2vec_format(att_embedding_path)

    query_embedding = get_embedding(query, use_weighted_embedding)
    if use_ABAE:
        
        abae_centers_embeddings = np.load('abae_centers.npy')

        embedding_path = os.path.join('w2v', "bible_word2vec_org")
        bible_wv = KeyedVectors.load_word2vec_format(embedding_path)
        # calculating cosine similarity with every aspects centers
        max_similarity = -1
        most_similar_aspect = 0

        for aspect in range(ASPECTS_NUM):
            cosine_similarity = 1 - spatial.distance.cosine(query_embedding, abae_centers_embeddings[aspect, ])
            # argmax cosine similarity
            if cosine_similarity > max_similarity:
                max_similarity = cosine_similarity  
                most_similar_aspect = aspect
        print(f'most similar aspect: {most_similar_aspect+1}')

        # get aspect of every verse
        verse2aspect_path = 'verse2aspect.npy'
        verse2aspect =  np.load(verse2aspect_path)
        
        aspects_qualified = (verse2aspect == most_similar_aspect)
        
        # verse2aspectprob_path ='aspects_probs.npy'
        # verse2aspectprob = np.load(verse2aspectprob_path)
        # bible_df['relevance'] = verse2aspectprob[:, most_similar_aspect] 
        qualified_verses = bible_df[aspects_qualified].copy()

        relevance = []
        for _, qualified_verse in qualified_verses.iterrows():
            # calculate cosine similarities for every verse
            qualified_verse_embedding = att_verse_vector[str(qualified_verse['id'])]
            relevance.append(1 - spatial.distance.cosine(query_embedding, qualified_verse_embedding))
        
        qualified_verses['relevance'] = relevance


        if sort_by_relevence:
            qualified_verses.sort_values(by=['relevance'], ascending = False, inplace = True)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_file = 'related_verses_ABAE.txt'
        output_path = os.path.join(output_folder, output_file)

        with open(output_path, 'w') as outfile:
            for _, verse_id in qualified_verses.iterrows():
                outfile.write('{book} {chapter}:{verse} "{content}"\n'.format(book=BOOK_NAME[int(verse_id['b'])-1], chapter=verse_id['c'],
                verse=verse_id['v'], content=verse_id['t']))
        print('Verses related to "{}" stored at ./{}'.format(query, output_path))

    else:
        ''' Use Vanilla Sentence Embedding '''
        similar_verses = [sent for sent, _ in verse_vector.similar_by_vector(query_embedding)]

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_file = 'related_verses_vanilla.txt'
        output_path = os.path.join(output_folder, output_file)

        with open(output_path, 'w') as outfile:
            for verse_id in similar_verses:
                verse = bible_df.loc[bible_df['id'] == int(verse_id)].iloc[0]
                outfile.write('{book} {chapter}:{verse} "{content}"\n'.format(book=BOOK_NAME[int(verse['b'])-1], chapter=verse['c'],
                verse=verse['v'], content=verse['t']))
        print('Verses related to "{}" stored at ./{}'.format(query, output_path))

if __name__ == '__main__':
    find_verse()