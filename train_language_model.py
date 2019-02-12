import logging

from sutra.data.datasets import get_dataset
from sutra.model.lm.transformer_lm import TransformerLanguageModelConfig
from sutra.model.lm.rnn_lm import RNNLanguageModelConfig
from sutra.model.lm.linear_lm import LinearLanguageModelConfig
from sutra.model.lm.trainer import train_language_model
from sutra.trainer import TrainingConfig
from sutra.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    model_type = 'transformer'

    if model_type == 'transformer':
        model_config = TransformerLanguageModelConfig(
            vocab_size=35000,
            seq_length=16,
            num_attention_heads=4,
            num_layers=4,
            embedding_size=128,
            encoding_size=128,
            dropout_prob=0.25)
    elif model_type == 'rnn':
        model_config = RNNLanguageModelConfig(
            vocab_size=35000,
            seq_length=16,
            embedding_size=128,
            encoding_size=128,
            dropout_prob=0.25)
    elif model_type == 'linear':
        model_config = LinearLanguageModelConfig(
            vocab_size=35000,
            seq_length=16,
            embedding_size=128,
            encoding_size=128,
            dropout_prob=0.25)

    batched_iterator = False

    training_config = TrainingConfig(
        device='cuda',
        epoch_length=8000,
        max_epochs=20,
        batch_size=256,
        optimizer='adam',
        learning_rate=0.001)

    dataset = get_dataset('wikitext-2', model_config.vocab_size)

    train_language_model(model_config, training_config, dataset, batched_iterator)


if __name__ == '__main__':
    main()
