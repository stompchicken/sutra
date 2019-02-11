import torch

from sutra.utils import setup_logging
from sutra.model.lm.rnn_lm import RNNLanguageModel, RNNLanguageModelConfig
from sutra.model.lm.transformer_lm import TransformerLanguageModel, TransformerLanguageModelConfig
from sutra.model.lm.linear_lm import LinearLanguageModel, LinearLanguageModelConfig
from sutra.model.lm.trainer import train_language_model
from sutra.trainer import TrainingConfig, Stage

from test.utils import create_batch
from test.asserts import assert_non_zero


def check_single_batch(model_config, batched_iterator):

    setup_logging()

    training_config = TrainingConfig(
        epoch_length=50,
        max_epochs=3,
        batch_size=4,
        optimizer='adam',
        learning_rate=0.01,
        device='cpu')

    class DummyDataset:
        train_data = list(range(20))
        valid_data = list(range(20))

    dataset = DummyDataset()

    trainer = train_language_model(model_config,
                                   training_config,
                                   dataset,
                                   batched_iterator)

    df = trainer.metrics.data
    print(df)
    df = df[df.stage == Stage.VALIDATION]
    df = df[df.epoch == training_config.max_epochs]
    loss = sum(df.loss) / len(df)

    assert loss < 0.1


def check_gradients(model, batched_iterator):
    batch_size = 2
    if batched_iterator:
        batch = create_batch([[0, 10],
                              [1, 11],
                              [2, 12],
                              [3, 13]],
                             [[1, 11],
                              [2, 12],
                              [3, 13],
                              [4, 14]])
    else:
        batch = create_batch([[0, 10],
                              [1, 11],
                              [2, 12],
                              [3, 13]],
                             [4, 14])

    state = model.init_state(batch_size)
    output, state = model(batch.data, state)
    cross_entropy = torch.nn.CrossEntropyLoss()
    loss = model.calculate_loss(output, batch.target, cross_entropy)
    assert loss != 0

    # Gradients are non-zero
    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert_non_zero(param.grad)


def test_linear_language_model():

    config = LinearLanguageModelConfig(
        vocab_size=24,
        seq_length=4,
        embedding_size=16,
        encoding_size=16,
        dropout_prob=0.0)

    model = LinearLanguageModel.from_config(config, 'cpu')

    check_gradients(model, batched_iterator=False)
    check_single_batch(config, batched_iterator=False)


def test_rnn_language_model():

    config = RNNLanguageModelConfig(
        vocab_size=24,
        seq_length=4,
        embedding_size=16,
        encoding_size=16,
        num_layers=1,
        dropout_prob=0.0)

    model = RNNLanguageModel.from_config(config, 'cpu')

    check_gradients(model, batched_iterator=True)
    check_single_batch(config, batched_iterator=True)


def test_transformer_language_model():
    config = TransformerLanguageModelConfig(
        vocab_size=24,
        seq_length=4,
        embedding_size=16,
        encoding_size=16,
        num_attention_heads=1,
        num_layers=2,
        dropout_prob=0.0)

    model = TransformerLanguageModel.from_config(config, 'cpu')

    check_gradients(model, batched_iterator=False)
    check_single_batch(config, batched_iterator=False)
