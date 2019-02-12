import argparse
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

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--log-experiment', action='store_true', help='')

    training = parser.add_argument_group('training')
    training.add_argument('--epoch-length', required=True, type=int, help='')
    training.add_argument('--max-epochs', required=True, type=int, help='')
    training.add_argument('--batch-size', required=True, type=int, help='')
    training.add_argument('--optimizer', required=True, type=str, choices=['adam'], help='')
    training.add_argument('--learning-rate', required=True, type=float, help='')
    training.add_argument('--device', required=True, type=str, help='')

    model = parser.add_argument_group('model')
    model.add_argument('--model', required=True, type=str, choices=['linear', 'rnn', 'transformer'],
                       help='')
    model.add_argument('--seq-length', required=True, type=int, help='')
    model.add_argument('--embedding-size', required=True, type=int, help='')
    model.add_argument('--dropout', required=True, type=float, help='')

    args = parser.parse_args()

    dataset_name = 'wikitext-2'
    vocab_size = 35000
    model_type = args.model

    if model_type == 'transformer':
        model_config = TransformerLanguageModelConfig(
            vocab_size=vocab_size,
            seq_length=args.seq_length,
            num_attention_heads=4,
            num_layers=2,
            embedding_size=args.embedding_size,
            encoding_size=args.embedding_size,
            dropout_prob=args.dropout)
        batched_iterator = False
    elif model_type == 'rnn':
        model_config = RNNLanguageModelConfig(
            vocab_size=vocab_size,
            seq_length=args.seq_length,
            embedding_size=args.embedding_size,
            encoding_size=args.embedding_size,
            dropout_prob=args.dropout)
        batched_iterator = True
    elif model_type == 'linear':
        model_config = LinearLanguageModelConfig(
            vocab_size=vocab_size,
            seq_length=args.seq_length,
            embedding_size=args.embedding_size,
            encoding_size=args.embedding_size,
            dropout_prob=args.dropout)
        batched_iterator = False

    training_config = TrainingConfig(
        epoch_length=args.epoch_length,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        device=args.device)
 
    dataset = get_dataset(dataset_name, model_config.vocab_size)

    train_language_model(model_config, training_config, dataset,
                         batched_iterator, args.log_experiment)


if __name__ == '__main__':
    main()
