from collections import Counter, deque

import numpy as np


class VocabBuilder(object):

    def __init__(self):
        self.vocab = Counter()

    def add_term(self, term):
        self.vocab[term] += 1

    def get_terms(self, size):
        return [x[0] for x in self.vocab.most_common()[:size]]


class Vocab(object):

    def __init__(self, vocab_builder, size):
        self.size = size
        self.tokens = ['__UNK__']
        self.tokens += vocab_builder.get_terms(size-1)
        self.token_index = {term: i for (i, term) in enumerate(self.tokens)}

    def term_to_index(self, term):
        return self.token_index.get(term, 0)

    def index_to_term(self, index):
        return self.tokens[index]


class Dataset(object):

    def __init__(self, path, vocab_size):
        self.tokens = []
        with open(path, 'r', encoding='utf8') as f:
            self.tokens.extend(self.tokenize(f.read()))

        vocab_builder = VocabBuilder()
        for token in self.tokens:
            vocab_builder.add_term(token)

        print('Building vocab: %d terms' % len(vocab_builder.vocab))

        self.vocab = Vocab(vocab_builder, vocab_size)
        print('Setting vocab to %d terms' % len(self.vocab.tokens))
        last_term = self.vocab.tokens[-1]
        print('Cutting off terms less than freq %d' % vocab_builder.vocab[last_term])

        self.mapped_tokens = np.array([self.vocab.term_to_index(token) for token in self.tokens])

    def batched_iterator(self, seq_length, batch_size):
        size = len(self.mapped_tokens) - seq_length + 1
        splits = np.arange(batch_size, size, batch_size)
        ngrams = np.array([self.mapped_tokens[i:i+seq_length] for i in range(0, size)])
        return np.array_split(ngrams, splits)

    def tokenize(self, text):
        return text.split()
