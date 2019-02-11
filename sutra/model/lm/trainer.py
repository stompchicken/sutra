import logging

import torch

from sutra.model.lm.transformer_lm import TransformerLanguageModel, TransformerLanguageModelConfig
from sutra.model.lm.linear_lm import LinearLanguageModel, LinearLanguageModelConfig
from sutra.model.lm.rnn_lm import RNNLanguageModel, RNNLanguageModelConfig

from sutra.data.iterators import LanguageModelIterator, BatchedLanguageModelIterator
from sutra.trainer import Trainer

logger = logging.getLogger(__name__)


def train_language_model(model_config, training_config,
                         dataset, batched_iterator):

    device = torch.device(training_config.device)

    if isinstance(model_config, TransformerLanguageModelConfig):
        model = TransformerLanguageModel.from_config(model_config, device)
    elif isinstance(model_config, LinearLanguageModelConfig):
        model = LinearLanguageModel.from_config(model_config, device)
    elif isinstance(model_config, RNNLanguageModelConfig):
        model = RNNLanguageModel.from_config(model_config, device)

    logger.info(model_config)

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

    iterator = BatchedLanguageModelIterator if batched_iterator else LanguageModelIterator
    train_iter = iterator(dataset.train_data,
                          training_config.batch_size,
                          model_config.seq_length)

    valid_iter = iterator(dataset.valid_data,
                          training_config.batch_size,
                          model_config.seq_length)

    trainer.train(train_iter, valid_iter)

    return trainer
