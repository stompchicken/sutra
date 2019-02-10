import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Transformer encoder, containing several stacks of encoding
    layers"""

    def __init__(self, embedding, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, data):
        seq_len, batch_size = data.size()

        data = self.embedding(data)
        for layer in self.layers:
            data = layer(data)
        return data


class EncoderLayer(nn.Module):
    """A single encoding layer, containing multi-headed attention and feed
    forward layers"""

    def __init__(self, encoding_size, self_attn, feed_forward, dropout_prob):
        super(EncoderLayer, self).__init__()
        self.encoding_size = encoding_size

        self.self_attn = self_attn
        self.layer_norm1 = nn.LayerNorm(encoding_size)
        self.dropout_layer1 = nn.Dropout(dropout_prob)

        self.feed_forward = feed_forward
        self.layer_norm2 = nn.LayerNorm(encoding_size)
        self.dropout_layer2 = nn.Dropout(dropout_prob)

    def forward(self, data):
        seq_len, batch_size, encoding_size = data.size()
        assert encoding_size == self.encoding_size

        attn = self.self_attn(data, data, data)
        data = self.layer_norm1(data + attn)
        data = self.dropout_layer1(data)

        ff = self.feed_forward(data)
        data = self.layer_norm2(data + ff)
        data = self.dropout_layer2(data)

        return data


class FeedForward(nn.Module):
    """Sublayer containing two fully connected feed forward networks"""

    def __init__(self, input_size, inner_size, dropout_prob):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(input_size, inner_size)
        self.w_2 = nn.Linear(inner_size, input_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(nn.Module):
    """Sublayer containing self attention network"""

    def __init__(self, input_size, num_heads, attention_fn, dropout_prob):
        super(MultiHeadedAttention, self).__init__()
        assert input_size % num_heads == 0
        self.input_size = input_size
        self.head_size = input_size // num_heads
        self.num_heads = num_heads
        self.attention_fn = attention_fn

        self.input_projections = nn.ModuleList(
            [nn.Linear(input_size, input_size) for _ in range(3)])
        self.output_projection = nn.Linear(input_size, input_size)

        self.dropout_fn = nn.Dropout(p=dropout_prob)

    def forward(self, query, key, value):
        input_dim = seq_len, batch_size, encoding_size = query.size()
        assert query.size() == key.size() == value.size()

        # Dimension of multi-headed attention tensor
        head_dim = (seq_len, batch_size, self.num_heads, self.head_size)

        query, key, value = self.project_input(query, key, value, head_dim)
        attn, _ = self.attention_fn(query, key, value,
                                    dropout_fn=self.dropout_fn)
        output = self.project_output(attn, input_dim)
        return output

    def project_input(self, query, key, value, dim):
        """Project input embeddings into multi-headed embedding space

        The projection into multiple 'heads' is done by a single
        linear layer
        """
        return (
            self.input_projections[0](query).view(*dim),
            self.input_projections[1](key).view(*dim),
            self.input_projections[2](value).view(*dim)
        )

    def project_output(self, data, dim):
        """Project multi-headed embeddings back to encoding size"""
        return self.output_projection(data.view(*dim))


class Embeddings(nn.Module):
    """Token embedding layer"""

    def __init__(self, vocab_size, embedding_size):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_position = 50

        self.token_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(self.max_position, embedding_size)
        self.positions = nn.Parameter(torch.LongTensor(range(self.max_position)),
                                      requires_grad=False)

    def forward(self, data):
        seq_length, batch_size = data.size()

        tokens = self.token_embeddings(data) * math.sqrt(self.embedding_size)

        positions = self.position_embeddings(self.positions[:seq_length])
        # Broadcast positional embeddings across entire batch
        positions = positions.unsqueeze(1).expand((seq_length, batch_size, self.embedding_size))

        return tokens + positions


def scaled_dot_attention(query, key, value, dropout_fn):
    """Applied attention to given values

    Returns:
        Tensor of same dimension as value
    """

    batch_size, seq_len, num_heads, head_size = query.size()
    assert query.size() == key.size() == value.size()

    scaling_factor = math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

    attn_potential = F.softmax(scores, dim=-1)
    attn_potential = dropout_fn(attn_potential)

    attn_output = torch.matmul(attn_potential, value)
    return attn_output, attn_potential


def create_encoder(vocab_size,
                   num_layers,
                   embedding_size,
                   encoding_size,
                   feed_forward_size,
                   num_attention_heads,
                   dropout_prob,
                   initialize=True):
    # Haven't really thought this through
    assert embedding_size == encoding_size

    attn = MultiHeadedAttention(encoding_size,
                                num_attention_heads,
                                scaled_dot_attention,
                                dropout_prob)
    ff = FeedForward(encoding_size, feed_forward_size, dropout_prob)

    embedding = Embeddings(vocab_size, embedding_size)

    encoder_layer = EncoderLayer(encoding_size, attn, ff, dropout_prob)
    encoder = Encoder(embedding, encoder_layer, num_layers)

    if initialize:
        for p in encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return encoder
