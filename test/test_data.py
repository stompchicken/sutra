import tempfile

from sutra.data import VocabBuilder, Vocab, Document, Corpus


def test_vocab_builder():
    vocab_builder = VocabBuilder()
    vocab_builder.add_term('A', count=3)
    vocab_builder.add_term('B', count=2)
    vocab_builder.add_term('C', count=1)

    assert vocab_builder.get_terms(3) == ['A', 'B', 'C']
    assert vocab_builder.get_terms(2) == ['A', 'B']
    assert vocab_builder.get_terms(1) == ['A']


def test_vocab():
    vocab = Vocab()
    vocab.add_term('A')
    vocab.add_term('B')
    vocab.add_term('C')

    assert len(vocab) == 4

    assert vocab.term_to_index('A') == 1
    assert vocab.term_to_index('B') == 2
    assert vocab.term_to_index('C') == 3
    assert vocab.term_to_index('D') == 0
    assert vocab.term_to_index('__UNK__') == 0

    assert vocab.index_to_term(0) == '__UNK__'
    assert vocab.index_to_term(1) == 'A'
    assert vocab.index_to_term(2) == 'B'
    assert vocab.index_to_term(3) == 'C'


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

    vocab = corpus.create_vocab(5)
    assert vocab.term_to_index('A') == 1
    assert vocab.term_to_index('B') == 2
    assert vocab.term_to_index('C') == 3
    assert vocab.index_to_term(1) == 'A'
    assert vocab.index_to_term(2) == 'B'
    assert vocab.index_to_term(3) == 'C'


def test_corpus_from_file():

    path = tempfile.mktemp()
    with open(path, 'w', encoding='utf8') as f:
        f.write("A A A B B C\n")
        f.write("C B A\n")

    corpus = Corpus.load_from_text(path)

    assert len(corpus) == 1
    assert corpus.documents[0].tokens == ['A', 'A', 'A', 'B', 'B', 'C', '<eos>',
                                          'C', 'B', 'A', '<eos>']
