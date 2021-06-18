import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_

class AttentionEncoder(nn.Module):
    """Segment encoder that produces segment vectors as the weighted
    average of word embeddings.
    """
    def __init__(self, vocab_size, emb_size, bias=True, M=None, b=None):
        """Initializes the encoder using a [vocab_size x emb_size] embedding
        matrix. The encoder learns a matrix M, which may be initialized
        explicitely or randomly.

        Parameters:
            vocab_size (int): the vocabulary size
            emb_size (int): dimensionality of embeddings
            bias (bool): whether or not to use a bias vector
            M (matrix): the attention matrix (None for random)
            b (vector): the attention bias vector (None for random)
        """
        super(AttentionEncoder, self).__init__()
        self.lookup = nn.Embedding(vocab_size, emb_size)
        self.M = nn.Parameter(torch.Tensor(emb_size, emb_size))
        if M is None:
            xavier_uniform_(self.M.data)
        else:
            self.M.data.copy_(M)
        if bias:
            self.b = nn.Parameter(torch.Tensor(1))
            if b is None:
                self.b.data.zero_()
            else:
                self.b.data.copy_(b)
        else:
            self.b = None

    def forward(self, inputs):
        """Forwards an input batch through the encoder"""
        x_wrd = self.lookup(inputs)
        x_avg = x_wrd.mean(dim=1)

        x = x_wrd.matmul(self.M)
        x = x.matmul(x_avg.unsqueeze(1).transpose(1,2))
        if self.b is not None:
            x += self.b

        x = torch.tanh(x)
        a = F.softmax(x, dim=1)

        z = a.transpose(1,2).matmul(x_wrd)
        z = z.squeeze()
        if z.dim() == 1:
            return z.unsqueeze(0)
        return z

    def set_word_embeddings(self, embeddings, fix_w_emb=True):
        """Initialized word embeddings dictionary and defines if it is trainable"""
        self.lookup.weight.data.copy_(embeddings)
        self.lookup.weight.requires_grad = not fix_w_emb

class AspectAutoencoder(nn.Module):
    """The aspect autoencoder class that defines our Multitask Aspect Extractor,
    but also implements the Aspect-Based Autoencoder (ABAE), if the aspect matrix
    not initialized using seed words
    """
    def __init__(self, vocab_size, emb_size, num_aspects=10, neg_samples=10,
            w_emb=None, a_emb=None, recon_method='centr', seed_w=None, num_seeds=None,
            attention=False, bias=True, M=None, b=None, fix_w_emb=True, fix_a_emb=False):
        """Initializes the autoencoder instance.

        Parameters:
            vocab_size (int): the vocabulary size
            emb_size (int): the embedding dimensionality
            num_aspects (int): the number of aspects
            neg_samples (int): the number of negative examples to use for the 
                               max-margin loss
            w_emb (matrix): a pre-trained embeddings matrix (None for random)
            a_emb (matrix): a pre-trained aspect matrix (None for random)
            recon_method (str): the segment reconstruction policy
                                - 'centr': uses centroid of seed words or single embeddings (ABAE)
                                - 'init': uses manually initialized seed weights
                                - 'fix': uses manually initialized seed weights, fixed during training
                                - 'cos': uses dynamic seed weights, obtained from cosine distance
            seed_w (matrix): seed weight matrix (for 'init' and 'fix')
            num_seeds (int): number of seed words
            attention (bool): use attention or not
            bias (bool): use bias vector for attention encoder
            M (matrix): matrix for attention encoder (optional)
            b (vector): bias vector for attention encoder (optional)
            fix_w_emb (bool): fix word embeddings throughout trainign
            fix_a_emb (bool): fix aspect embeddings throughout trainign
        """
        super(AspectAutoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.recon_method = recon_method
        self.num_seeds = num_seeds
        self.attention = attention
        self.bias = bias
        self.num_aspects = num_aspects
        self.neg_samples = neg_samples

        if not attention:
            self.seg_encoder = nn.EmbeddingBag(vocab_size, emb_size)
        else:
            self.seg_encoder = AttentionEncoder(vocab_size, emb_size, bias, M, b)
        self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)

        if w_emb is None:
            xavier_uniform_(self.seg_encoder.weight.data)
        else:
            assert w_emb.size() == (vocab_size, emb_size), "Word embedding matrix has incorrect size"
            if not attention:
                self.seg_encoder.weight.data.copy_(w_emb)
                self.seg_encoder.weight.requires_grad = not fix_w_emb
            else:
                self.seg_encoder.set_word_embeddings(w_emb, fix_w_emb)

        if a_emb is None:
            self.a_emb = nn.Parameter(torch.Tensor(num_aspects, emb_size))
            xavier_uniform_(self.a_emb.data)
        else:
            assert a_emb.size()[0] == num_aspects and a_emb.size()[-1] == emb_size, "Aspect embedding matrix has incorrect size"
            self.a_emb = nn.Parameter(torch.Tensor(a_emb.size()))
            self.a_emb.data.copy_(a_emb)
            self.a_emb.requires_grad = not fix_a_emb

        if recon_method == 'fix':
            self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
            self.seed_w.data.copy_(seed_w)
            self.seed_w.requires_grad = False
        elif recon_method == 'init':
            self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
            self.seed_w.data.copy_(seed_w)
        else:
            self.seed_w = None

        self.lin = nn.Linear(emb_size, num_aspects)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, batch_num=None):
        if self.training:
            # mask used for randomly selected negative examples
            self.cur_mask = self._create_neg_mask(inputs.size(0))

        if not self.attention:
            offsets = Variable(torch.arange(0, inputs.numel(), inputs.size(1), out=inputs.data.new().long()))
            enc = self.seg_encoder(inputs.view(-1), offsets)
        else:
            enc = self.seg_encoder(inputs)
            self.enc = enc

        x = self.lin(enc)
        a_probs = self.softmax(x)

        if self.recon_method == 'centr':
            # r = a_probs.matmul(self.a_emb)
            r = a_probs.matmul(F.normalize(self.a_emb, dim=1))

        elif self.recon_method == 'fix':
            a_emb_w = self.a_emb.mul(self.seed_w.view(self.num_aspects, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_aspects, 1, 1).mul(a_emb_w).sum(dim=2).sum(dim=1)
        elif self.recon_method == 'init':
            seed_w_norm = F.softmax(self.seed_w, dim=1)
            a_emb_w = self.a_emb.mul(seed_w_norm.view(self.num_aspects, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_aspects, 1, 1).mul(a_emb_w).sum(dim=2).sum(dim=1)
        elif self.recon_method == 'cos':
            sim = F.cosine_similarity(enc.unsqueeze(1),
                    self.a_emb.view(1, self.num_aspects*self.num_seeds, self.emb_size),
                    dim=2).view(-1, self.num_aspects, self.num_seeds)
            self.seed_w = F.softmax(sim, dim=2)
            a_emb_w = self.a_emb.mul(self.seed_w.view(-1, self.num_aspects, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_aspects, 1, 1).mul(a_emb_w).sum(dim=2).sum(dim=1)

        return r, a_probs

    def _create_neg_mask(self, batch_size):
        """Creates a mask for randomly selecting negative samples"""
        multi_weights = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        neg = min(batch_size - 1, self.neg_samples)

        mask = torch.multinomial(multi_weights, neg)
        mask = mask.unsqueeze(2).expand(batch_size, neg, self.emb_size)
        mask = Variable(mask, requires_grad=False)
        return mask

    def set_targets(self, module, input, output):
        """Sets positive and negative samples"""
        assert self.cur_mask is not None, 'Tried to set targets without a mask'
        batch_size = output.size(0)

        if torch.cuda.is_available():
            mask = self.cur_mask.cuda()
        else:
            mask = self.cur_mask

        self.negative = Variable(output.data).expand(batch_size, batch_size, self.emb_size).gather(1, mask)
        self.positive = Variable(output.data)
        self.cur_mask = None

    def get_targets(self):
        assert self.positive is not None, 'Positive targets not set; needs a forward pass first'
        assert self.negative is not None, 'Negative targets not set; needs a forward pass first'
        return self.positive, self.negative

    def get_aspects(self):
        if self.a_emb.dim() == 2:
            return self.a_emb
        else:
            return self.a_emb.mean(dim=1)

    def train(self, mode=True):
        super(AspectAutoencoder,  self).train(mode)
        if self.encoder_hook is None:
            self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)
        return self

    def eval(self):
        super(AspectAutoencoder, self).eval()
        if self.encoder_hook is not None:
            self.encoder_hook.remove()
            self.encoder_hook = None
        return self
