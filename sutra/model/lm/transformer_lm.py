import logging
import typing

import torch
import torch.nn as nn

from sutra.model.transformer import create_encoder

logger = logging.getLogger(__name__)


class TransformerLanguageModelConfig(typing.NamedTuple):
    vocab_size: int
    seq_length: int
    embedding_size: int
    encoding_size: int
    num_attention_heads: int
    num_layers: int
    dropout_prob: float


# TODO: Better handling of droput config
class TransformerLanguageModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 encoding_size,
                 num_attention_heads,
                 num_layers,
                 dropout_prob,
                 device):
        super(TransformerLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.device = device

        self.encoder = create_encoder(vocab_size=vocab_size,
                                      num_layers=num_layers,
                                      embedding_size=embedding_size,
                                      encoding_size=encoding_size,
                                      feed_forward_size=256,
                                      num_attention_heads=num_attention_heads,
                                      dropout_prob=dropout_prob)
        self.encoder.to(device)

        self.decoder = torch.nn.Linear(encoding_size, vocab_size).to(device)
        self.decoder.weight = self.encoder.embedding.token_embeddings.weight

    def config(self):
        return {
            "name": self.__class__.__module__ + '.' + self.__class__.__qualname__,
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "encoding_size": self.encoding_size,
            "num_attention_heads": self.num_attention_heads,
            "device": str(self.device)
        }

    def init_state(self, batch_size):
        return None

    def repackage_state(self, state):
        pass

    def forward(self, data, state):
        seq_length, batch_size = data.size()
        # Take encoding of last element in the sequence
        encodings = self.encoder.forward(data)[-1, :]
        # Tied embeddings
        decoded = self.decoder(encodings)

        return decoded, state

    def calculate_loss(self, output, target, loss_fn):
        return loss_fn(output, target)
