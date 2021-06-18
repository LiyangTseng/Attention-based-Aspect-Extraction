import os
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import BibleDataset, custom_collate_fn
from train_wv import get_wv
from tqdm import tqdm

class VanillaSetenceEmbedding(nn.Module):
    ''' Generating sentence embeddding by averaging word embedding ''' 
    def __init__(self, vocab_size, emb_size):
        super(VanillaSetenceEmbedding, self).__init__()
        self.lookup = nn.Embedding(vocab_size, emb_size)
        
    def forward(self, inputs):
        x_wrd = self.lookup(inputs)
        x_avg = x_wrd.mean(dim=1)
        return x_avg

def main():
    dataset = BibleDataset("./t_kjv.csv", word_to_index=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=custom_collate_fn)

    verses = [verse_tuple[0] for verse_tuple in dataset]
    bible_kjv_sents = []
    for verse in verses:
        verse_list = [dataset.id2word[verse_idx] for verse_idx in verse]
        bible_kjv_sents.append(verse_list)

    bible_wv = get_wv(bible_kjv_sents)
    embed_size = bible_wv.vector_size
    vocab_size = len(dataset.word2id.keys())
    net = VanillaSetenceEmbedding(vocab_size, embed_size).cuda()

    w2v_folder = 'w2v'
    if not os.path.exists(w2v_folder):
        os.mkdir(w2v_folder)
    embedding_path = os.path.join(w2v_folder, "bible_verse_org")
    vocab_path = os.path.join(w2v_folder, "bible_verse_vocalbulary")

    min_len = 1
    for lines, id in tqdm(dataloader, position=0, leave=False, desc='Forward Propagation'):
        inputs = Variable(torch.Tensor(lines).long())

        if inputs.shape[1] < min_len:
            continue
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        verse_embeddings = net(inputs)
        verse_vector = KeyedVectors(vector_size=embed_size)
        verse_vector.add(list(id), verse_embeddings.detach().cpu().numpy())
        verse_vector.save_word2vec_format(embedding_path, vocab_path)
        print('Bible verse embeddings generated and stored at {}'.format(embedding_path))
        

if __name__ == '__main__':
    main()