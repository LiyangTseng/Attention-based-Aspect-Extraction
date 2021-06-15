import warnings
warnings.filterwarnings("ignore")
import re
import csv
from nltk.corpus import stopwords
from flair.data import Sentence
from torch.utils.data import Dataset

class BibleDataset(Dataset):
    def __init__(self, path, word_to_index=True):
        self.word_to_index = word_to_index
        self.vocab_size = 0
        self.word2id = dict()
        self.id2word = dict()

        self.stop_words = set(stopwords.words("english"))

        self.data = list(self.reader(path))

    def reader(self, path):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                line = tokenize(row['t'], self.stop_words)

                if self.word_to_index:
                    line = self.tokens_to_index(line)
                    if len(line) == 0: continue
                    line = padding(line, 100)

                yield line, row["id"]

    def tokens_to_index(self, tokens):
        ids = list()
        for token in tokens:
            if token not in self.word2id:
                self.word2id[token] = self.vocab_size
                self.id2word[self.vocab_size] = token
                self.vocab_size += 1
            ids.append(self.word2id[token])
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    return list(zip(*batch))


def tokenize(sentence, stop_words):
    sentence = clean_str(sentence)
    tokens = [token.text for token in Sentence(sentence).tokens]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def padding(li, length):
    while len(li) < length: li.extend(li)
    return li[:length]

def clean_str(text):
    text = text.lower()

    # replace all numbers with 0
    text = re.sub(r"[-+]?[-/.\d]*[\d]+[:,.\d]*", ' 0 ', text)

    # English-specific pre-processing
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)

    # remove commas, colons, semicolons, periods, brackets, hyphens, <, >, and quotation marks
    text = re.sub(r'[,:;\.\(\)-/"<>]', " ", text)

    # separate exclamation marks and question marks
    text = re.sub(r"!+", " ! ", text)
    text = re.sub(r"\?+", " ? ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()