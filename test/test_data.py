from sutra.data.data import Document, Corpus, VocabBuilder, Vocab


def test_vocab_builder():
    vocab_builder = VocabBuilder()
    vocab_builder.add_term('A', count=3)
    vocab_builder.add_term('B', count=2)
    vocab_builder.add_term('C', count=1)

    assert len(vocab_builder) == 4

    assert vocab_builder.get_terms(5) == ['<unk>', 'A', 'B', 'C']
    assert vocab_builder.get_terms(4) == ['<unk>', 'A', 'B', 'C']
    assert vocab_builder.get_terms(3) == ['<unk>', 'A', 'B']
    assert vocab_builder.get_terms(2) == ['<unk>', 'A']
    assert vocab_builder.get_terms(1) == ['<unk>']

    # Adding special terms is a no-op
    vocab_builder.add_term('<unk>')
    assert len(vocab_builder) == 4


def test_vocab_builder_ordering():
    vocab_builder = VocabBuilder()
    vocab_builder.add_term('C')
    vocab_builder.add_term('B')
    vocab_builder.add_term('A')

    assert len(vocab_builder) == 4
    # Vocabulary is ordered first by frequency then alphabetically
    assert vocab_builder.get_terms(4) == ['<unk>', 'A', 'B', 'C']


def test_vocab():
    vocab = Vocab()
    vocab.add_term('A')
    vocab.add_term('B')
    vocab.add_term('C')

    assert len(vocab) == 3

    assert vocab.term_to_index('A') == 0 and vocab.index_to_term(0) == 'A'
    assert vocab.term_to_index('B') == 1 and vocab.index_to_term(1) == 'B'
    assert vocab.term_to_index('C') == 2 and vocab.index_to_term(2) == 'C'

    assert vocab.term_to_index('D') == 0


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
