import collections

import torch
import torch.nn as nn

import logging


class RNNEncoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 encoding_size,
                 num_layers,
                 dropout_prob,
                 device):
        super(RNNEncoder, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_size).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)
        self.num_layers = num_layers

        # Dropout is not meaningful for single-later RNNs
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
        embeddings = self.embedding(input.to(self.device))
        embeddings = self.dropout(embeddings)
        output, hidden = self.rnn(embeddings, hidden)
        output = self.dropout(output)
        return output, hidden


class RNNLanguageModel(nn.Module):

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

        self.encoder = RNNEncoder(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  encoding_size=encoding_size,
                                  num_layers=num_layers,
                                  dropout_prob=dropout_prob,
                                  device=device)
        self.decoder = nn.Linear(encoding_size, vocab_size).to(device)

        if embedding_size == encoding_size:
            self.decoder.weight = self.encoder.embedding.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        return self.encoder.init_hidden(batch_size)

    def forward(self, input, hidden):
        encoding, hidden = self.encoder.forward(input, hidden)
        decoded = self.decoder(encoding.view(encoding.size(0) * encoding.size(1), encoding.size(2)))
        return decoded.view(encoding.size(0), encoding.size(1), decoded.size(1)), hidden


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their
    history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


RNNLanguageModelConfig = collections.namedtuple(
    'RNNLanguageModelConfig',
    ['vocab_size', 'batch_size', 'seq_length', 'embedding_size',
     'encoding_size', 'dropout_prob'])
