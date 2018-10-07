import logging
import collections


logger = logging.getLogger(__name__)


class VocabBuilder(object):

    def __init__(self):
        self.vocab = collections.Counter()

    def add_term(self, term, count=1):
        self.vocab[term] += count

    def get_terms(self, size):
        return [x[0] for x in self.vocab.most_common()[:size]]


class Vocab(object):

    def __init__(self):
        self.terms = ['__UNK__']
        self.term_index = {}

    def __len__(self):
        return len(self.terms)

    def add_term(self, term):
        self.term_index[term] = len(self.terms)
        self.terms.append(term)

    def term_to_index(self, term):
        return self.term_index.get(term, 0)

    def index_to_term(self, index):
        return self.terms[index]

    @staticmethod
    def from_vocab_builder(vocab_builder, size):
        vocab = Vocab()
        for term in vocab_builder.get_terms(size - 1):
            vocab.add_term(term)

        return vocab


Document = collections.namedtuple('Document', ['tokens'])


class Corpus(object):

    def __init__(self):
        self.documents = []

    def __len__(self):
        return len(self.documents)

    def add_document(self, document):
        self.documents.append(document)

    @staticmethod
    def load_from_text(path):
        corpus = Corpus()
        with open(path, 'r', encoding='utf8') as f:
            tokens = []
            for line in f:
                tokens.extend(Corpus.tokenize(line))
                tokens.append('<eos>')
            corpus.documents.append(Document(tokens=tokens))
        return corpus

    @staticmethod
    def tokenize(text):
        return text.split()

    def create_vocab(self, vocab_size):
        vocab_builder = VocabBuilder()
        for document in self.documents:
            for token in document.tokens:
                vocab_builder.add_term(token)

        return Vocab.from_vocab_builder(vocab_builder, vocab_size)
