import collections
import logging
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim

import utils
import language_model as lm


logger = logging.getLogger(__name__)


class RNNEncoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 encoding_size,
                 num_layers,
                 dropout_prob,
                 device):
        super(RNNEncoder, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_size).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)
        self.num_layers = num_layers

        # Dropout is not meaningful for single-later RNNs
        rnn_dropout_prob = 0.0 if self.num_layers == 1 else dropout_prob
        self.rnn = nn.LSTM(self.embedding_size,
                           self.encoding_size,
                           self.num_layers,
                           dropout=rnn_dropout_prob).to(device)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        size = (self.num_layers, batch_size, self.encoding_size)
        return (torch.zeros(*size, device=self.device),
                torch.zeros(*size, device=self.device))

    def forward(self, input, hidden):
        embeddings = self.embedding(input.to(self.device))
        embeddings = self.dropout(embeddings)
        output, hidden = self.rnn(embeddings, hidden)
        output = self.dropout(output)
        return output, hidden


class RNNLanguageModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 encoding_size,
                 num_layers,
                 dropout_prob,
                 device):
        super(RNNLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size

        self.encoder = RNNEncoder(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  encoding_size=encoding_size,
                                  num_layers=num_layers,
                                  dropout_prob=dropout_prob,
                                  device=device)
        self.decoder = nn.Linear(encoding_size, vocab_size).to(device)

        # Tied weights
        if embedding_size == encoding_size:
            self.decoder.weight = self.encoder.embedding.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        return self.encoder.init_hidden(batch_size)

    def forward(self, input, hidden):
        encoding, hidden = self.encoder.forward(input, hidden)
        decoded = self.decoder(encoding.view(encoding.size(0) * encoding.size(1), encoding.size(2)))
        return decoded.view(encoding.size(0), encoding.size(1), decoded.size(1)), hidden


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their
    history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


RNNLanguageModelConfig = collections.namedtuple(
    'RNNLanguageModelConfig',
    ['vocab_size', 'batch_size', 'seq_length', 'embedding_size',
     'encoding_size', 'dropout_prob'])


def train(config, device, max_epochs):
    logging.info("Device: %s" % device)
    logging.info("Config: %s" % str(config))
    logging.info("Max epochs: %d" % max_epochs)

    train_iter, valid_iter, test_iter = lm.load_wikitext2(
        config.batch_size, config.seq_length, config.vocab_size, device)

    model = RNNLanguageModel(vocab_size=config.vocab_size,
                             embedding_size=config.embedding_size,
                             encoding_size=config.encoding_size,
                             num_layers=2,
                             dropout_prob=config.dropout_prob,
                             device=device)

    def evaluate(dataset):
        model.eval()
        hidden = model.init_hidden(config.batch_size)
        cross_entropy = nn.CrossEntropyLoss()
        metrics = {
            'cross_entropy': utils.Metric('cross_entropy'),
            'perplexity': utils.Metric('perplexity')
        }

        for batch in dataset:
            hidden = repackage_hidden(hidden)
            output, hidden = model(batch.text, hidden)

            # Reshape into flat tensors
            predictions = output.view(-1, config.vocab_size)
            targets = batch.target.view(-1)

            loss = cross_entropy(predictions, targets)

            metrics['cross_entropy'].update(loss.item())
            metrics['perplexity'].update(math.exp(loss.item()))

        return metrics

    def train(dataset, optimizer, criterion):
        model.train()
        hidden = model.init_hidden(config.batch_size)
        loss_estimate = utils.Metric('loss')
        perplexity = utils.Metric('ppl')
        tokens_per_second = utils.Metric('tokens/s')

        epoch = dataset.epoch
        for batch in dataset:
            if dataset.epoch > epoch:
                break

            start = time.time()

            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()

            output, hidden = model(batch.text, hidden)

            # Reshape into flat tensors
            predictions = output.view(-1, config.vocab_size)
            targets = batch.target.view(-1)

            loss = criterion(predictions, targets)
            loss.backward()

            loss_estimate.update(loss.item())
            perplexity.update(math.exp(loss.item()))

            grad_clip = 5.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            end = time.time()
            tokens_per_second.update(
                float(config.batch_size * config.seq_length) / (end-start))

            if dataset.iterations % 100 == 0:
                logger.debug('Training [%d.%d]: %s %s %s' %
                             (dataset.epoch, dataset.iterations,
                              loss_estimate, perplexity, tokens_per_second))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     patience=4,
                                                     verbose=True)
    early_stopping = utils.EarlyStopping(6)

    for i in range(max_epochs):
        logger.info('Epoch: %d' % i)
        train(train_iter, optimizer, criterion)
        torch.save(model.state_dict(), 'rnn_lm.model')

        metrics = evaluate(valid_iter)
        logger.info('Validation: %s' % ', '.join(
            [str(m) for m in metrics.values()]))

        scheduler.step(metrics['cross_entropy'].get_estimate())
        early_stopping.add_value(metrics['cross_entropy'].get_estimate())
        if early_stopping.should_stop():
            logging.info("Early stopping")
            break

    metrics = evaluate(test_iter)
    logger.info('Test: %s' % ', '.join([str(m) for m in metrics.values()]))


def main():
    utils.setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RNNLanguageModelConfig(
        vocab_size=30000,
        batch_size=64,
        seq_length=35,
        embedding_size=650,
        encoding_size=650,
        dropout_prob=0.5)
    train(config, device, 40)


if __name__ == '__main__':
    main()
