import pytest

from sutra.data.data import Document, Corpus, VocabBuilder, Vocab, UNKNOWN_TERM


def test_vocab_builder():
    vocab_builder = VocabBuilder(specials=[])
    vocab_builder.add_term('A')
    vocab_builder.add_term('B')
    vocab_builder.add_term('C')

    assert len(vocab_builder) == 3
    assert len(vocab_builder.get_terms()) == 3

    assert vocab_builder.get_terms() == ['A', 'B', 'C']

    assert vocab_builder.get_terms(4) == ['A', 'B', 'C']
    assert vocab_builder.get_terms(3) == ['A', 'B', 'C']
    assert vocab_builder.get_terms(2) == ['A', 'B']
    assert vocab_builder.get_terms(1) == ['A']
    assert vocab_builder.get_terms(0) == []


def test_vocab_builder_specials():
    vocab_builder = VocabBuilder(specials=['1', '2', '3'])
    assert vocab_builder.get_terms() == ['1', '2', '3']

    # Adding a special is a no-op
    vocab_builder.add_term('3')
    assert vocab_builder.get_terms() == ['1', '2', '3']

    # Specials always come first in the vocab
    vocab_builder.add_term('A', count=100)
    assert vocab_builder.get_terms() == ['1', '2', '3', 'A']


def test_vocab_builder_count_ordering():
    vocab_builder = VocabBuilder(specials=[])
    vocab_builder.add_term('C', count=2)
    vocab_builder.add_term('B', count=1)
    vocab_builder.add_term('A', count=1)

    # Vocabulary is ordered first by frequency then alphabetically
    assert vocab_builder.get_terms() == ['C', 'A', 'B']


def test_vocab_default_unk():
    vocab_builder = VocabBuilder()
    assert vocab_builder.get_terms() == [UNKNOWN_TERM]


def test_vocab():
    vocab = Vocab()
    vocab.add_term('A')
    vocab.add_term('B')
    vocab.add_term('C')

    assert len(vocab) == 3

    assert vocab.term_to_index('A') == 0 and vocab.index_to_term(0) == 'A'
    assert vocab.term_to_index('B') == 1 and vocab.index_to_term(1) == 'B'
    assert vocab.term_to_index('C') == 2 and vocab.index_to_term(2) == 'C'


def test_vocab_oov():
    vocab = Vocab()
    vocab.add_term('A')

    # Out-of-vocab terms get mapped to zero
    assert vocab.term_to_index('B') == 0


def test_vocab_duplicates():
    vocab = Vocab()
    vocab.add_term('A')

    with pytest.raises(ValueError):
        vocab.add_term('A')

    assert len(vocab) == 1


def test_vocab_from_builder():

    vocab_builder = VocabBuilder()
    vocab_builder.add_term('A', count=3)
    vocab_builder.add_term('B', count=2)
    vocab_builder.add_term('C', count=1)

    vocab = Vocab.from_vocab_builder(vocab_builder, 10)
    assert len(vocab) == 4
    assert vocab.term_to_index('A') == 1
    assert vocab.term_to_index('B') == 2
    assert vocab.term_to_index('C') == 3
    assert vocab.index_to_term(1) == 'A'
    assert vocab.index_to_term(2) == 'B'
    assert vocab.index_to_term(3) == 'C'

    vocab = Vocab.from_vocab_builder(vocab_builder, 2)
    assert len(vocab) == 2
    assert vocab.term_to_index('A') == 1
    assert vocab.index_to_term(1) == 'A'
    assert vocab.term_to_index('B') == 0


def test_corpus_vocab():
    tokens = "A A A B B C".split()

    corpus = Corpus()
    corpus.add_document(Document(tokens))

    assert len(corpus) == 1

    vocab = corpus.create_vocab()

    assert vocab.terms == [UNKNOWN_TERM, 'A', 'B', 'C']

    assert vocab.term_to_index(UNKNOWN_TERM) == 0
    assert vocab.term_to_index('A') == 1
    assert vocab.term_to_index('B') == 2
    assert vocab.term_to_index('C') == 3
    assert vocab.index_to_term(0) == UNKNOWN_TERM
    assert vocab.index_to_term(1) == 'A'
    assert vocab.index_to_term(2) == 'B'
    assert vocab.index_to_term(3) == 'C'
