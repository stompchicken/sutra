import pytest

import sutra.data.datasets as datasets
import sutra.data.iterators as iterators

from test.asserts import assert_eq


def test_wikitext2():
    cache = datasets.DatasetCache()
    wikitext = datasets.WikiText2(cache, vocab_size=50000)

    # Numbers taken from the Wikitext website
    assert len(wikitext.vocab) == 33278
    assert len(wikitext.train_data) == 2088628
    assert len(wikitext.valid_data) == 217646
    assert len(wikitext.test_data) == 245569


@pytest.mark.skip(reason="Too slow! and torchtext dependency")
def test_torchtext_equivalence():
    import torchtext
    cache = datasets.DatasetCache()
    wikitext = datasets.WikiText2(cache,
                                  vocab_size=50000,
                                  lowercase=True,
                                  specials=['<unk>', '<pad>'])

    TEXT = torchtext.data.Field(lower=True, batch_first=True)
    train, valid, test = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train, max_size=50000)

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
                                                pad_to_batch_size=True)

    valid_it2 = iterators.LanguageModelIterator(wikitext.valid_data,
                                                batch_size,
                                                seq_length,
                                                pad_to_batch_size=True)

    test_it2 = iterators.LanguageModelIterator(wikitext.test_data,
                                               batch_size,
                                               seq_length,
                                               pad_to_batch_size=True)

    for batch1, batch2 in zip(train_it, train_it2):
        assert_eq(batch1.text, batch2.text)
        assert_eq(batch1.target, batch2.target)

    for batch1, batch2 in zip(valid_it, valid_it2):
        assert_eq(batch1.text, batch2.text)
        assert_eq(batch1.target, batch2.target)

    for batch1, batch2 in zip(test_it, test_it2):
        assert_eq(batch1.text, batch2.text)
        assert_eq(batch1.target, batch2.target)
