import logging
from collections import Counter, deque

import numpy as np


logger = logging.getLogger(__name__)


class VocabBuilder(object):

    def __init__(self):
        self.vocab = Counter()

    def add_term(self, term):
        self.vocab[term] += 1

    def get_terms(self, size):
        return [x[0] for x in self.vocab.most_common()[:size]]


class Vocab(object):

    def __init__(self, vocab_builder, size):
        logger.info('Building vocab: %d terms (from %d)' % (size, len(vocab_builder.vocab)))
        self.size = size
        self.tokens = ['__UNK__']
        self.tokens += vocab_builder.get_terms(size-1)
        self.token_index = {term: i for (i, term) in enumerate(self.tokens)}

        last_term = self.tokens[-1]
        logging.info('Cutting off terms less than freq %d' % vocab_builder.vocab[last_term])

    def term_to_index(self, term):
        return self.token_index.get(term, 0)

    def index_to_term(self, index):
        return self.tokens[index]


class Dataset(object):

    def __init__(self, tokens, vocab):
        self.vocab = vocab
        self.tokens = np.array([self.vocab.term_to_index(token) for token in tokens])

    def ngrams(self, seq_length):
        size = len(self.tokens) - (seq_length - 1)
        return np.array([self.tokens[i:i+seq_length] for i in range(0, size)])

    def batched_iterator(self, seq_length, batch_size):
        ngrams = self.ngrams(seq_length)
        num_batches = len(ngrams) // batch_size
        if num_batches == 0:
            return []
        else:
            return np.split(ngrams[:num_batches*batch_size], num_batches)


def tokenize(text):
    return text.split()


def create_dataset(path, vocab):
    tokens = []
    with open(path, 'r', encoding='utf8') as f:
        tokens.extend(tokenize(f.read()))
    dataset = Dataset(tokens, vocab)
    return dataset


def create_vocab(path, vocab_size):
    vocab_builder = VocabBuilder()
    with open(path, 'r', encoding='utf8') as f:
        for token in tokenize(f.read()):
            vocab_builder.add_term(token)

    return Vocab(vocab_builder, vocab_size)
