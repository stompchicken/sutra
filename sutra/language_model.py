import math
import logging
import torchtext


logger = logging.getLogger(__name__)


def load_text(text, vocab_size):
    TEXT = torchtext.data.Field(lower=True, batch_first=True)
    example = torchtext.data.Example.fromlist([text], [('text', TEXT)])
    dataset = torchtext.data.Dataset([example], {'text': TEXT})

    TEXT.build_vocab(dataset, max_size=vocab_size - 2)

    return dataset


def load_wikitext2(vocab_size):
    TEXT = torchtext.data.Field(lower=True, batch_first=True)

    # make splits for data
    train, valid, test = torchtext.datasets.WikiText2.splits(TEXT, device='cpu')

    logger.info("Train: %d tokens" % len(train[0].text))
    logger.info("Valid: %d tokens" % len(valid[0].text))
    logger.info("Test:  %d tokens" % len(test[0].text))

    # build the vocabulary
    TEXT.build_vocab(train, max_size=vocab_size - 2)
    logger.info("Vocab: %d terms" % len(TEXT.vocab))

    return train, valid, test


def bptt_iterator(train, valid, test, batch_size, seq_length, device):
    return torchtext.data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        bptt_len=seq_length,
        device=device)


class NgramIterator(torchtext.data.Iterator):
    def __init__(self, dataset, batch_size, seq_length, **kwargs):
        self.seq_length = seq_length
        super(NgramIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        n = len(self.dataset[0].text) // (self.batch_size - 1)
        return n - self.seq_length

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None

        length = math.floor(len(text) // self.batch_size) * self.batch_size
        text = text[:length]

        data = TEXT.numericalize([text], device=self.device)

        data = data.view(self.batch_size, -1).t().contiguous()
        dataset = torchtext.data.Dataset(examples=self.dataset.examples,
                                         fields=[('text', TEXT), ('target', TEXT)])

        while True:
            for i in range(0, len(self) - self.seq_length):
                self.iterations += 1
                yield torchtext.data.Batch.fromvars(dataset, self.batch_size,
                                                    text=data[i:i + self.seq_length],
                                                    target=data[i + self.seq_length])
            if not self.repeat:
                return
            else:
                self.epochs += 1


def ngram_iterator(train, valid, test, batch_size, seq_length, device):
    return NgramIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        seq_length=seq_length,
        device=device)


def iterator(train, valid, test, batch_size, device):
    return torchtext.data.BucketIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        sort_key=None,
        device=device,
        sort_within_batch=None)
