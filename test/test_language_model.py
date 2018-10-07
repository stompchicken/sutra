import torch

from sutra.data import Corpus, Document, Vocab
from sutra.language_model import LanguageModellingDataset
from test.asserts import assert_eq


def test_dataset():
    dataset = LanguageModellingDataset(3)
    dataset.add_tokens(range(10))

    assert len(dataset) == 4
    assert_eq(dataset[0], [0, 1, 2])
    assert_eq(dataset[1], [3, 4, 5])
    assert_eq(dataset[2], [6, 7, 8])
    assert_eq(dataset[3], [9])


def test_dataset_from_corpus():
    tokens = 'We broke our backs lifing Moloch to heaven'.split()

    corpus = Corpus()
    corpus.add_document(Document(tokens=tokens))

    vocab = Vocab()
    for token in tokens:
        vocab.add_term(token)

    dataset = LanguageModellingDataset.create_from_corpus(corpus, vocab, 4)
    assert len(dataset) == 2
    assert_eq(dataset[0], [1, 2, 3, 4])
    assert_eq(dataset[1], [5, 6, 7, 8])


def test_dataset_batching():
    dataset = LanguageModellingDataset(3)
    dataset.add_tokens(range(10))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3)

    it = dataloader.__iter__()
    print(next(it))
    print(next(it))
    assert False
