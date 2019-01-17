import pytest
from numpy.testing import assert_array_equal

import sutra.data.data as data
import sutra.data.iterators as iterators
import sutra.data.datasets as datasets


@pytest.mark.skip(reason="Too slow")
def test_torchtext_equivalence():
    import torchtext
    cache = datasets.DatasetCache()
    wikitext = datasets.WikiText2(cache, vocab_size=50000, lowercase=True)

    TEXT = torchtext.data.Field(lower=True, batch_first=True)
    train, valid, test = torchtext.datasets.WikiText2.splits(TEXT)

    # text equivalence
    assert train[0].text == wikitext.train_tokens
    assert valid[0].text == wikitext.valid_tokens
    assert test[0].text == wikitext.test_tokens

    # vocab equivalence
    vocab_size = 50000
    TEXT.build_vocab(train, max_size=vocab_size - 2)
    assert train.fields['text'].vocab.itos == wikitext.vocab.terms

    # data equivalence
    batch_size = 3
    seq_length = 5
    train_it, valid_it, test_it = torchtext.data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        bptt_len=seq_length)

    train_it2 = iterators.LanguageModelIterator(wikitext.train_data,
                                                batch_size,
                                                seq_length,
                                                pad_last_batch=True)

    valid_it2 = iterators.LanguageModelIterator(wikitext.valid_data,
                                                batch_size,
                                                seq_length,
                                                pad_last_batch=True)

    test_it2 = iterators.LanguageModelIterator(wikitext.test_data,
                                               batch_size,
                                               seq_length,
                                               pad_last_batch=True)

    for batch1, batch2 in zip(train_it, train_it2):
        assert_array_equal(batch1.text, batch2.data)
        assert_array_equal(batch1.target, batch2.target)

    for batch1, batch2 in zip(valid_it, valid_it2):
        assert_array_equal(batch1.text, batch2.data)
        assert_array_equal(batch1.target, batch2.target)

    for batch1, batch2 in zip(test_it, test_it2):
        assert_array_equal(batch1.text, batch2.data)
        assert_array_equal(batch1.target, batch2.target)


def assert_lm_iterator(input_data, batch_data, batch_target,
                       batch_size, seq_length):
    expected = [data.Batch(d, t) for d, t in zip(batch_data, batch_target)]

    batches = list(iterators.LanguageModelIterator(
        input_data,
        batch_size=batch_size,
        seq_length=seq_length,
        partial_batch=False))
    assert len(batches) == len(expected) - 1

    for i in range(len(batches)):
        assert_array_equal(expected[i].data, batches[i].data)
        assert_array_equal(expected[i].target, batches[i].target)

    batches = list(iterators.LanguageModelIterator(
        input_data,
        batch_size=batch_size,
        seq_length=seq_length,
        partial_batch=True))
    assert len(batches) == len(expected)

    for i in range(len(batches)):
        assert_array_equal(expected[i].data, batches[i].data)
        assert_array_equal(expected[i].target, batches[i].target)


def test_lm_iterator():

    data = [
        [[0, 6, 12],
         [1, 7, 13]],
        [[2, 8, 14],
         [3, 9, 15]],
        [[4, 10, 16]]
    ]

    target = [
        [[1, 7, 13],
         [2, 8, 14]],
        [[3, 9, 15],
         [4, 10, 16]],
        [[5, 11, 17]],
    ]

    assert_lm_iterator(list(range(20)),
                       data,
                       target,
                       batch_size=3,
                       seq_length=2)

    data = [
        [[0, 6, 12],
         [1, 7, 13],
         [2, 8, 14]],
        [[3, 9, 15],
         [4, 10, 16]]
    ]

    target = [
        [[1, 7, 13],
         [2, 8, 14],
         [3, 9, 15]],
        [[4, 10, 16],
         [5, 11, 17]],
    ]

    assert_lm_iterator(list(range(20)),
                       data,
                       target,
                       batch_size=3,
                       seq_length=3)

    data = [
        [[0, 10],
         [1, 11],
         [2, 12],
         [3, 13]],
        [[4, 14],
         [5, 15],
         [6, 16],
         [7, 17]],
        [[8, 18]],
    ]

    target = [
        [[1, 11],
         [2, 12],
         [3, 13],
         [4, 14]],
        [[5, 15],
         [6, 16],
         [7, 17],
         [8, 18]],
        [[9, 19]]
    ]

    assert_lm_iterator(list(range(20)),
                       data,
                       target,
                       batch_size=2,
                       seq_length=4)
