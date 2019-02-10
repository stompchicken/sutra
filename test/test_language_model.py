import torch

from sutra.utils import setup_logging
from sutra.data.iterators import LanguageModelIterator, BatchedLanguageModelIterator
from sutra.model.lm.rnn_lm import RNNLanguageModel
from sutra.model.lm.transformer_lm import TransformerLanguageModel
from sutra.model.lm.linear_lm import LinearLanguageModel
from sutra.trainer import TrainingConfig, Trainer, Stage

from test.utils import create_batch
from test.asserts import assert_non_zero


def check_single_batch(model, batched_iterator):

    setup_logging()

    # Hack for now
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    cross_entropy = torch.nn.CrossEntropyLoss()

    def train_fn(batch, state):
        hidden = model.repackage_state(state)
        output, hidden = model(batch.text, hidden)
        loss = model.calculate_loss(output, batch.target, cross_entropy)

        return {
            "loss": loss
        }, hidden

    training_config = TrainingConfig(
        epoch_length=50,
        max_epochs=3,
        batch_size=4,
        optimizer=None)

    trainer = Trainer(training_config, model, train_fn, train_fn, optimizer)

    if batched_iterator:
        train_iter = BatchedLanguageModelIterator(list(range(20)), 4, 4)
        valid_iter = BatchedLanguageModelIterator(list(range(20)), 4, 4)
    else:
        train_iter = LanguageModelIterator(list(range(20)), 4, 4)
        valid_iter = LanguageModelIterator(list(range(20)), 4, 4)

    trainer.train(train_iter, valid_iter)

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
    model = LinearLanguageModel(
        vocab_size=24,
        seq_length=4,
        embedding_size=16,
        encoding_size=16,
        dropout_prob=0.0,
        device='cpu')

    check_gradients(model, batched_iterator=False)
    check_single_batch(model, batched_iterator=False)


def test_rnn_language_model():
    model = RNNLanguageModel(
        vocab_size=24,
        embedding_size=16,
        encoding_size=16,
        num_layers=1,
        dropout_prob=0.0,
        device='cpu')

    check_gradients(model, batched_iterator=True)
    check_single_batch(model, batched_iterator=True)


def test_transformer_language_model():
    model = TransformerLanguageModel(
        vocab_size=24,
        embedding_size=16,
        encoding_size=16,
        num_attention_heads=1,
        num_layers=2,
        dropout_prob=0.0,
        device='cpu')

    check_gradients(model, batched_iterator=False)
    check_single_batch(model, batched_iterator=False)
