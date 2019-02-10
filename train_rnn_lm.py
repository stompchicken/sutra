import logging

import torch

from sutra.model.lm.rnn_lm import RNNLanguageModel, RNNLanguageModelConfig
from sutra.data.iterators import BatchedLanguageModelIterator
from sutra.data.datasets import DatasetCache, WikiText2
from sutra.trainer import Trainer, TrainingConfig
from sutra.utils import setup_logging
from sutra.memory import profiler

logger = logging.getLogger(__name__)


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
        hidden = model.repackage_state(state)
        output, hidden = model(batch.text.to(device), hidden)
        loss = model.calculate_loss(output, batch.target.to(device), cross_entropy)

        return {
            "loss": loss
        }, hidden

    trainer = Trainer(training_config, model, train_fn, train_fn, optimizer,
                      log_experiment=False)

    seq_length = model_config.seq_length

    cache = DatasetCache()
    wikitext = WikiText2(cache, vocab_size=model_config.vocab_size)
    train_iter = BatchedLanguageModelIterator(wikitext.train_data,
                                              training_config.batch_size,
                                              seq_length,
                                              device)

    valid_iter = BatchedLanguageModelIterator(wikitext.valid_data,
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
        vocab_size=35000,
        seq_length=16,
        num_layers=1,
        embedding_size=256,
        encoding_size=256,
        dropout_prob=0.25)

    training_config = TrainingConfig(
        epoch_length=2000,
        max_epochs=20,
        batch_size=64,
        optimizer=None)

    trainer(model_config, training_config, device)


if __name__ == '__main__':
    main()
