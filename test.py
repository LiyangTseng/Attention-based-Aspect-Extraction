import sys
import os
import numpy as np
from scipy import spatial
import pandas as pd
from embedding import get_embedding

output_folder = 'output'
output_file = 'related_verses.txt'
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
    python test.py [query]
    '''
    query = sys.argv[1]
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

    # get aspect of every verse
    verse2aspect_path = 'verse2aspect.npy'
    verse2aspect =  np.load(verse2aspect_path)

    bible_verses_path = "./t_kjv.csv"
    bible_df = pd.read_csv(bible_verses_path)

    aspects_qualified = (verse2aspect == most_similar_aspect)
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_path = os.path.join(output_folder, output_file)
    
    with open(output_path, 'w') as outfile:
        for _, verse in bible_df[aspects_qualified].iterrows():
            outfile.write('{book} {chapter}:{verse} "{content}"\n'.format(book=BOOK_NAME[int(verse['b'])-1], chapter=verse['c'],
            verse=verse['v'], content=verse['t']))
    print('Verses related to "{}" stored at ./{}'.format(query, output_path))

if __name__ == '__main__':
    find_verse()