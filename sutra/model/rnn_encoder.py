import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RNNEncoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 encoding_size,
                 num_layers,
                 dropout_prob,
                 device):
        super(RNNEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_size).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)
        self.num_layers = num_layers

        # Dropout is not meaningful for single-layer RNNs
        if self.num_layers == 1 and dropout_prob > 0:
            logger.warning("Non-zero dropout for single-layer RNN will have no effect")
        rnn_dropout_prob = 0.0 if self.num_layers == 1 else dropout_prob

        self.rnn = nn.LSTM(self.embedding_size,
                           self.encoding_size,
                           self.num_layers,
                           dropout=rnn_dropout_prob).to(device)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        size = (self.num_layers, batch_size, self.encoding_size)
        return (torch.zeros(*size, device=self.device),
                torch.zeros(*size, device=self.device))

    def forward(self, input, hidden):
        seq_len, batch_size = input.size()
        assert hidden[0].size(1) == batch_size

        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)

        output, hidden = self.rnn(embeddings, hidden)
        output = self.dropout(output)

        return output, hidden
