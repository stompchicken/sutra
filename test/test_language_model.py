import pytest
import torch

from sutra.utils import setup_logging
from sutra.data.iterators import LanguageModelIterator
from sutra.model.lm.rnn_lm import RNNLanguageModel
from sutra.trainer import TrainingConfig, Trainer, Stage

from test.utils import create_batch
from test.asserts import assert_non_zero


@pytest.mark.skip("Slow")
def test_language_model():

    setup_logging()
    device = 'cpu'
    vocab_size = 100
    model = RNNLanguageModel(
        vocab_size=vocab_size,
        embedding_size=16,
        encoding_size=16,
        num_layers=1,
        dropout_prob=0.0,
        device=device)

    # Hack for now
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    cross_entropy = torch.nn.CrossEntropyLoss()

    def train_fn(batch, state):
        hidden = model.repackage_state(state)
        output, hidden = model(batch.text.to(device), hidden)

        # Reshape into flat tensors
        predictions = output.view(-1, vocab_size)
        targets = batch.target.view(-1).to(device)

        loss = cross_entropy(predictions, targets)

        return {
            "loss": loss
        }, hidden

    def eval_fn(batch, state):
        hidden = model.repackage_state(state)
        output, hidden = model(batch.text.to(device), hidden)

        # Reshape into flat tensors
        predictions = output.view(-1, vocab_size)
        targets = batch.target.view(-1).to(device)

        loss = cross_entropy(predictions, targets)

        return {
            "loss": loss
        }, hidden

    training_config = TrainingConfig(
        epoch_length=100,
        max_epochs=5,
        batch_size=4,
        optimizer=None)

    trainer = Trainer(training_config, model, train_fn, eval_fn, optimizer)

    train_iter = LanguageModelIterator(list(range(20)), 4, 4)
    valid_iter = LanguageModelIterator(list(range(20)), 4, 4)

    trainer.train(train_iter, valid_iter)

    df = trainer.metrics.data
    df = df[df.stage == Stage.VALIDATION]
    df = df[df.epoch == training_config.max_epochs]
    loss = sum(df.loss) / len(df)

    assert loss < 0.01


def test_model():

    device = 'cpu'
    vocab_size = 100
    model = RNNLanguageModel(
        vocab_size=vocab_size,
        embedding_size=16,
        encoding_size=16,
        num_layers=1,
        dropout_prob=0.0,
        device=device)

    batch_size = 2
    batch = create_batch([[0, 10],
                          [1, 11],
                          [2, 12],
                          [3, 13]],
                         [[1, 11],
                          [2, 12],
                          [3, 13],
                          [4, 14]])

    state = model.init_state(batch_size)
    output, state = model(batch.text.to(device), state)
    cross_entropy = torch.nn.CrossEntropyLoss()
    loss = model.calculate_loss(output, batch.target.to(device), cross_entropy)
    assert loss > 0

    # Gradients are non-zero
    loss.backward()
    for param in model.parameters():
        assert_non_zero(param.grad)
