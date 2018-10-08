import math
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext

from utils import Metric, EarlyStopping

import rnn_lm

logger = logging.getLogger(__name__)

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] %(name)-20s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def load_wikitext2(batch_size, seq_length, vocab_size):
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
        device='cuda')


def train(config, device, max_epochs):
    logging.info("Device: %s" % device)
    logging.info("Config: %s" % str(config))
    logging.info("Max epochs: %d" % max_epochs)

    train_iter, valid_iter, test_iter = load_wikitext2(
        config.batch_size, config.seq_length, config.vocab_size)

    model = rnn_lm.RNNLanguageModel(vocab_size=config.vocab_size,
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
            'cross_entropy': Metric('cross_entropy'),
            'perplexity': Metric('perplexity')
        }

        for batch in dataset:
            hidden = rnn_lm.repackage_hidden(hidden)
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
        loss_estimate = Metric('loss')
        perplexity = Metric('ppl')
        tokens_per_second = Metric('tokens/s')

        epoch = dataset.epoch
        for batch in dataset:
            if dataset.epoch > epoch:
                break

            start = time.time()

            hidden = rnn_lm.repackage_hidden(hidden)
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
    early_stopping = EarlyStopping(6)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = rnn_lm.RNNLanguageModelConfig(
        vocab_size=30000,
        batch_size=64,
        seq_length=35,
        embedding_size=650,
        encoding_size=650,
        dropout_prob=0.5)
    train(config, device, 40)


if __name__ == '__main__':
    main()
