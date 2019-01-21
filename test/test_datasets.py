import sutra.data.datasets as datasets


def test_wikitext2():
    cache = datasets.DatasetCache()
    wikitext = datasets.WikiText2(cache, vocab_size=50000)

    # Numbers taken from the Wikitext website
    assert len(wikitext.vocab) == 33278
    assert len(wikitext.train_data) == 2088628
    assert len(wikitext.valid_data) == 217646
    assert len(wikitext.test_data) == 245569
