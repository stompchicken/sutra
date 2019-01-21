import logging

import torch

from sutra.model.lm.rnn_lm import RNNLanguageModel, RNNLanguageModelConfig
from sutra.data.iterators import LanguageModelIterator
from sutra.data.datasets import DatasetCache, WikiText2
from sutra.trainer import Trainer, TrainingConfig
from sutra.utils import setup_logging
from sutra.memory import profiler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their
    history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def trainer(model_config, training_config, device):

    model = RNNLanguageModel(
        vocab_size=model_config.vocab_size,
        embedding_size=model_config.embedding_size,
        encoding_size=model_config.encoding_size,
        num_layers=model_config.num_layers,
        dropout_prob=model_config.dropout_prob,
        device=device)

    # Hack for now
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    cross_entropy = torch.nn.CrossEntropyLoss()

    def train_fn(batch, state):
        hidden = repackage_hidden(state)
        output, hidden = model(batch.text.to(device), hidden)

        # Reshape into flat tensors
        predictions = output.view(-1, model_config.vocab_size)
        targets = batch.target.view(-1).to(device)

        loss = cross_entropy(predictions, targets)

        return {
            "loss": loss
        }, hidden

    def eval_fn(batch, state):
        hidden = repackage_hidden(state)
        output, hidden = model(batch.text.to(device), hidden)

        # Reshape into flat tensors
        predictions = output.view(-1, model_config.vocab_size)
        targets = batch.target.view(-1).to(device)

        loss = cross_entropy(predictions, targets)

        return {
            "loss": loss
        }, hidden

    trainer = Trainer(training_config, model, train_fn, eval_fn, optimizer)

    seq_length = model_config.seq_length

    cache = DatasetCache()
    wikitext = WikiText2(cache, vocab_size=model_config.vocab_size)
    train_iter = LanguageModelIterator(wikitext.train_data,
                                       training_config.batch_size,
                                       seq_length,
                                       device,
                                       repeat=True)

    valid_iter = LanguageModelIterator(wikitext.valid_data,
                                       training_config.batch_size,
                                       seq_length,
                                       device)

    profiler.memory_usage("Before training")
    trainer.train(train_iter, valid_iter)


def main():
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device=%s", device)
    model_config = RNNLanguageModelConfig(
        vocab_size=50000,
        seq_length=15,
        num_layers=2,
        embedding_size=320,
        encoding_size=320,
        dropout_prob=0.5)

    training_config = TrainingConfig(
        epoch_length=1000,
        max_epochs=5,
        batch_size=128,
        optimizer=None)

    trainer(model_config, training_config, device)


if __name__ == '__main__':
    main()
