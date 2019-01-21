import typing
import logging

import torch

from sutra.model.rnn_encoder import RNNEncoder

logger = logging.getLogger(__name__)


class RNNLanguageModelConfig(typing.NamedTuple):
    vocab_size: int
    seq_length: int
    num_layers: int
    embedding_size: int
    encoding_size: int
    dropout_prob: float


class RNNLanguageModel(torch.nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 encoding_size,
                 num_layers,
                 dropout_prob,
                 device):
        super(RNNLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size
        self.num_layers = num_layers
        self.device = device

        self.encoder = RNNEncoder(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  encoding_size=encoding_size,
                                  num_layers=num_layers,
                                  dropout_prob=dropout_prob,
                                  device=device)
        self.decoder = torch.nn.Linear(encoding_size, vocab_size).to(device)

        # Tied input/output embeddings
        if embedding_size == encoding_size:
            self.decoder.weight = self.encoder.embedding.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_state(self, batch_size):
        return self.encoder.init_hidden(batch_size)

    def forward(self, input, hidden):
        encoding, hidden = self.encoder.forward(input, hidden)

        seq_length, batch_size, encoding_size = encoding.size()
        assert encoding_size == self.encoding_size

        decoded = self.decoder(encoding.view(seq_length * batch_size,
                                             encoding_size))
        vocab_size = decoded.size(1)

        return decoded.view(seq_length, batch_size, vocab_size), hidden
