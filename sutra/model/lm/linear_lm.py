import typing
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LinearLanguageModelConfig(typing.NamedTuple):
    vocab_size: int
    seq_length: int
    embedding_size: int
    encoding_size: int
    dropout_prob: float


class LinearLanguageModel(torch.nn.Module):
    """Language models based on stacked fully connected layers
    Based on Bengio et al, 2006. "Neural probabilistic language models"
    """

    def __init__(self,
                 vocab_size,
                 seq_length,
                 embedding_size,
                 encoding_size,
                 dropout_prob,
                 device):
        super(LinearLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size
        self.dropout_prob = dropout_prob
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_size).to(device)

        context_embedding_size = embedding_size * seq_length
        self.linear1 = torch.nn.Linear(context_embedding_size, encoding_size).to(device)
        self.linear2 = torch.nn.Linear(encoding_size, encoding_size).to(device)
        self.decoder = torch.nn.Linear(encoding_size, vocab_size).to(device)

        self.dropout = nn.Dropout(dropout_prob).to(device)

    @classmethod
    def from_config(cls, config, device):
        return cls(**config._asdict(), device=device)

    def config(self):
        return {
            "name": self.__class__.__module__ + '.' + self.__class__.__qualname__,
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "encoding_size": self.encoding_size,
            "dropout_prob": self.dropout_prob,
            "device": str(self.device)
        }

    def init_state(self, batch_size):
        return None

    def repackage_state(self, state):
        pass

    def forward(self, data, hidden):
        embeddings = self.embedding.forward(data)

        # Concatenate embeddings
        seq_length, batch_size, embedding_size = embeddings.size()
        context_dim = (batch_size, seq_length * embedding_size)
        context_embedding = embeddings.permute(1, 0, 2).contiguous().view(context_dim)

        encodings = F.relu(self.linear1(context_embedding))
        encodings = self.dropout(encodings)

        encodings = F.relu(self.linear2(encodings))
        encodings = self.dropout(encodings)

        decoded = self.decoder(encodings)
        return decoded, None

    def calculate_loss(self, output, target, loss_fn):
        return loss_fn(output, target)
