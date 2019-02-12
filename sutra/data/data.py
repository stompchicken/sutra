import logging
import collections
import typing
import torch

logger = logging.getLogger(__name__)

UNKNOWN_TERM = '<unk>'


class VocabBuilder:
    """Utility class for building vocabularies

    Stores terms and counts and returns a list of terms ranked by
    count, with special handling for certain reserved symbols.
    """

    def __init__(self, specials=None):
        self.vocab = collections.defaultdict(int)
        self.specials = [UNKNOWN_TERM] if specials is None else specials

    def add_term(self, term, count=1):
        if term not in self.specials:
            self.vocab[term] += count

    def __len__(self):
        return len(self.specials) + len(self.vocab)

    def get_terms(self, size=None):
        """Return a ranked list of terms

        The list always contains the special terms first (in the order
        they were given to the constructor) and after that the added
        terms ranked by count

        Args:
            size: the length of the ranked list

        """
        size = len(self) if size is None else size
        sorted_terms = sorted([(-f, t) for (t, f) in self.vocab.items()])
        num_terms = min(len(sorted_terms), size - len(self.specials))
        return self.specials + [x[1] for x in sorted_terms[:num_terms]]


class Vocab:
    """A mapping of terms to indexes"""

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
        """Constuct a vocabulary from a VocabBuilder"""

        vocab = Vocab()

        for term in vocab_builder.get_terms(size):
            vocab.add_term(term)

        return vocab


class Document(typing.NamedTuple):
    tokens: typing.List[str]


class Corpus:
    """A collection of documents

    The main purpose of a corpus is to contain documenta and generate
    a vocabulary.
    """

    def __init__(self, documents=None):
        self.documents = documents if documents is not None else []

    def __len__(self):
        return len(self.documents)

    def add_document(self, document):
        self.documents.append(document)

    def create_vocab(self, vocab_size=None, specials=None):
        self.vocab_builder = VocabBuilder(specials=specials)
        for document in self.documents:
            for token in document.tokens:
                self.vocab_builder.add_term(token)

        documents = len(self.documents)
        tokens = sum(len(document.tokens) for document in self.documents)
        unique_tokens = len(self.vocab_builder)
        logger.info(f"Building vocab from: {documents} documents, {tokens} tokens, {unique_tokens} unique tokens")

        vocab = Vocab.from_vocab_builder(self.vocab_builder, vocab_size)
        logger.info(f"Created vocab with: {len(vocab)} tokens")

        return vocab


class Batch(typing.NamedTuple):
    data: torch.Tensor
    target: torch.Tensor
