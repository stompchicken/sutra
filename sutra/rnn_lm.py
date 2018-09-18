import math
import copy

import torch
import torch.nn as nn
import torch.autograd as auto
import torch.optim as optim
import torch.nn.functional as F

import language_model as lm

import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(name)-20s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class Metric(object):

    def __init__(self, name):
        self.name = name
        self.mean = 0.0
        self.updates = 1

    def update(self, metric):
        self.mean = self.mean + ((metric - self.mean) / self.updates)
        self.updates += 1

    def reset(self):
        self.mean = 0
        self.updates = 1

    def get_estimate(self):
        return self.mean

    def __repr__(self):
        return '%s: %.5f' % (self.name, self.mean)

class RNNEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, encoding_size, dropout_prob, device):
        super(RNNEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.encoding_size = encoding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)
        self.num_layers = 1
        self.rnn = nn.LSTM(self.embedding_size, self.encoding_size, self.num_layers,
                           dropout=dropout_prob).to(device)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.encoding_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.encoding_size, device=device))

    def forward(self, input, hidden):
        embeddings = self.dropout(self.embedding(input))
        embeddings = embeddings.permute(1,0,2)
        output, hidden = self.rnn(embeddings, hidden)
        output = self.dropout(output)
        return output, hidden


class RNNLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, encoding_size, dropout_prob, device):
        super(RNNLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size

        self.encoder = RNNEncoder(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  encoding_size=encoding_size,
                                  dropout_prob=dropout_prob,
                                  device=device)
        self.decoder = nn.Linear(encoding_size, vocab_size).to(device)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size, device):
        return self.encoder.init_hidden(batch_size, device)

    def forward(self, input, hidden):
        encoding, hidden = self.encoder.forward(input, hidden)
        decoded = self.decoder(encoding.view(encoding.size(0)*encoding.size(1), encoding.size(2)))
        return decoded.view(encoding.size(0), encoding.size(1), decoded.size(1)), hidden


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print("Device: %s" % device)

    vocab_size = 20000
    embedding_size = 64
    encoding_size = 64
    dropout_prob = 0.0

    train_file = 'data/language_modelling/wikitext-2/wiki.train.tokens'
    valid_file = 'data/language_modelling/wikitext-2/wiki.valid.tokens'
    test_file = 'data/language_modelling/wikitext-2/wiki.test.tokens'

    vocab = lm.create_vocab(train_file, vocab_size)

    train_data = lm.create_dataset(train_file, vocab)
    valid_data = lm.create_dataset(valid_file, vocab)
    test_data = lm.create_dataset(test_file, vocab)

    model = RNNLanguageModel(vocab_size,
                             embedding_size=embedding_size,
                             encoding_size=encoding_size,
                             dropout_prob=dropout_prob,
                             device=device)
    model.train()

    batch_size = 64
    seq_length = 15

    logger.info('Batch size: %d' % batch_size)
    logger.info('Sequence length: %d' % seq_length)
    logger.info('Corpus size: %d' % len(train_data.tokens))


    def evaluate(dataset):
        hidden = model.init_hidden(batch_size, device)
        cross_entropy = nn.CrossEntropyLoss()
        metrics = {
            'cross_entropy': Metric('cross_entropy'),
            'perplexity': Metric('perplexity')
        }

        it = dataset.batched_iterator(seq_length=seq_length, batch_size=batch_size)
        for i, batch in enumerate(it):
            context = torch.from_numpy(batch[:, :-1]).to(device)
            target = torch.from_numpy(batch[:, -1]).to(device)

            output, hidden = model(context, hidden)

            value = cross_entropy(output[-1], target).item()

            metrics['cross_entropy'].update(value)

        metrics['perplexity'].update(2 ** metrics['cross_entropy'].get_estimate())

        return metrics


    def train(dataset, optimizer, criterion):
        hidden = model.init_hidden(batch_size, device)
        loss_estimate = Metric('loss')

        it = dataset.batched_iterator(seq_length=seq_length, batch_size=batch_size)
        for i, batch in enumerate(it):
            context = torch.from_numpy(batch[:, :-1]).to(device)
            target = torch.from_numpy(batch[:, -1]).to(device)

            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()

            output, hidden = model(context, hidden)
            loss = criterion(output[-1], target)

            loss.backward()

            grad_clip = 5.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            loss_estimate.update(loss.item())

            optimizer.step()

            if i % 10000 == 0:
                logger.debug('Iteration: %d %s' % (i, loss_estimate))

        logger.info(loss_estimate)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    for i in range(1):
        logger.info('Epoch: %d' % i)
        train(train_data, optimizer, criterion)

        metrics = evaluate(valid_data)
        logger.info('Validation: %s' % ', '.join([str(m) for m in metrics.values()]))


if __name__ == '__main__':
    main()
