# Attention-based Aspect Extraction (ABAE)

## Prerequisite
- run <code> pip install -r requirements.txt</code> to setup the environment

## How to run
- <code>train_wv.py</code> train word embeddings from dataset
- <code> train.ipynb </code>: detailed procedure of training ABAE model
- <code>python test.py [query]</code> will find the most related verses to query, the reuslts via ABAE will be stored at ./output/related_verses_ABAE.txt
## Main File Structure
<pre>
.
|-- ABAE_structure.png
|-- README.md
|-- abae_centers.npy
|-- aspects_probs.npy
|-- dataset.py
|-- embedding.py
|-- generate_verse_embedding.py
|-- loss.py
|-- model.py
|-- output
|   |-- related_verses_ABAE.txt
|   |-- related_verses_vanilla.txt
|-- requirements.txt
|-- t_kjv.csv
|-- test.py
|-- train.ipynb
|-- train_wv.py
|-- verse2aspect.npy
`-- w2v
    |-- bible_verse_att_org
    |-- bible_verse_att_vocalbulary
    |-- bible_verse_org
    |-- bible_verse_vocalbulary
    |-- bible_word2vec_org
    `-- bible_word2vec_vocalbulary
</pre>

## ABAE Structure
<img src="ABAE_structure.png" width="600"/>
