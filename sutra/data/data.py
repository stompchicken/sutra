import logging
import collections
import typing
import numpy as np

logger = logging.getLogger(__name__)


class VocabBuilder:

    def __init__(self, specials=None):
        self.vocab = collections.defaultdict(int)
        self.specials = specials or ['<unk>']

    def add_term(self, term, count=1):
        if term not in self.specials:
            self.vocab[term] += count

    def __len__(self):
        return len(self.specials) + len(self.vocab)

    def get_terms(self, size):
        sorted_terms = sorted([(-f, t) for (t, f) in self.vocab.items()])
        num_terms = min(len(sorted_terms), size - len(self.specials))
        return self.specials + [x[1] for x in sorted_terms[:num_terms]]


class Vocab:

    def __init__(self):
        self.terms = []
        self.term_index = {}

    def __len__(self):
        return len(self.terms)

    def add_term(self, term):
        if term not in self.term_index:
            self.term_index[term] = len(self.terms)
            self.terms.append(term)
        else:
            raise ValueError('Attempted to insert term \'{}\' into vocab ' +
                             'when it was already present'.format(term))

    def term_to_index(self, term):
        return self.term_index.get(term, 0)

    def index_to_term(self, index):
        return self.terms[index]

    @staticmethod
    def from_vocab_builder(vocab_builder, size=None):
        vocab = Vocab()
        size = size or len(vocab_builder.vocab)

        for term in vocab_builder.get_terms(size):
            vocab.add_term(term)

        return vocab


class Document(typing.NamedTuple):
    tokens: typing.List[str]


class Corpus(object):

    def __init__(self, documents=None):
        self.documents = documents or []

    def __len__(self):
        return len(self.documents)

    def add_document(self, document):
        self.documents.append(document)

    def create_vocab(self, vocab_size=None, specials=None):
        vocab_builder = VocabBuilder(specials=specials)
        for document in self.documents:
            for token in document.tokens:
                vocab_builder.add_term(token)

        return Vocab.from_vocab_builder(vocab_builder, vocab_size)


class Batch(typing.NamedTuple):
    data: np.array
    target: np.array
