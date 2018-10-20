import logging
import torchtext


logger = logging.getLogger(__name__)


def load_wikitext2(batch_size, seq_length, vocab_size, device):
    TEXT = torchtext.data.Field(lower=True, batch_first=True)

    # make splits for data
    train, valid, test = torchtext.datasets.WikiText2.splits(TEXT)

    logger.info("Train: %d tokens" % len(train[0].text))
    logger.info("Valid: %d tokens" % len(valid[0].text))
    logger.info("Test:  %d tokens" % len(test[0].text))

    # build the vocabulary
    TEXT.build_vocab(train, max_size=vocab_size - 2)
    logger.info("Vocab: %d terms" % len(TEXT.vocab))

    # make iterator for splits
    return torchtext.data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        bptt_len=seq_length,
        device=device)
