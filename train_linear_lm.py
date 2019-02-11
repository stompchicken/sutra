import logging

import torch

from sutra.model.lm.linear_lm import LinearLanguageModel, LinearLanguageModelConfig
from sutra.data.iterators import LanguageModelIterator
from sutra.data.datasets import DatasetCache, WikiText2
from sutra.trainer import Trainer, TrainingConfig
from sutra.utils import setup_logging
from sutra.memory import profiler

logger = logging.getLogger(__name__)


def trainer(model_config, training_config, device):

    model = LinearLanguageModel(
        vocab_size=model_config.vocab_size,
        seq_length=model_config.seq_length,
        embedding_size=model_config.embedding_size,
        encoding_size=model_config.encoding_size,
        dropout_prob=model_config.dropout_prob,
        device=device)

    cross_entropy = torch.nn.CrossEntropyLoss()

    def train_fn(batch, state):
        hidden = model.repackage_state(state)
        output, hidden = model(batch.text.to(device), hidden)
        loss = model.calculate_loss(output, batch.target.to(device), cross_entropy)

        return {
            "loss": loss
        }, hidden

    trainer = Trainer(training_config, model, train_fn, train_fn,
                      log_experiment=False)

    seq_length = model_config.seq_length

    cache = DatasetCache()
    wikitext = WikiText2(cache, vocab_size=model_config.vocab_size)
    train_iter = LanguageModelIterator(wikitext.train_data,
                                       training_config.batch_size,
                                       seq_length,
                                       device)

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
    model_config = LinearLanguageModelConfig(
        vocab_size=35000,
        seq_length=16,
        embedding_size=128,
        encoding_size=128,
        dropout_prob=0.25)

    training_config = TrainingConfig(
        epoch_length=8000,
        max_epochs=20,
        batch_size=256,
        optimizer='adam',
        learning_rate=0.001)

    trainer(model_config, training_config, device)


if __name__ == '__main__':
    main()
