import math
import copy

import torch
import torch.nn as nn
import torch.autograd as auto
import torch.nn.functional as F


class Encoder(nn.Module):
    """Transformer encoder, containing several stacks of encoding
    layers"""

    def __init__(self, embedding, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderLayer(nn.Module):
    """A single encoding layer, containing multi-headed attention and feed
    forward layers"""

    def __init__(self, size, self_attn, feed_forward, dropout_ratio):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.layer_norm1 = nn.LayerNorm(size)
        self.dropout_layer1 = nn.Dropout(dropout_ratio)

        self.feed_forward = feed_forward
        self.layer_norm2 = nn.LayerNorm(size)
        self.dropout_layer2 = nn.Dropout(dropout_ratio)

    def forward(self, x):
        attn = self.self_attn(x, x, x)
        x = self.layer_norm1(x + attn)
        x = self.dropout_layer1(x)

        ff = self.feed_forward(x)
        x = self.layer_norm2(x + ff)
        x = self.dropout_layer2(x)

        return x


class FeedForward(nn.Module):
    """Sublayer containing two fully connected feed forward networks"""

    def __init__(self, input_size, inner_size, dropout_ratio=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(input_size, inner_size)
        self.w_2 = nn.Linear(inner_size, input_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(nn.Module):
    """Sublayer containing self attention network"""

    def __init__(self, input_size, num_heads, attention_fn, dropout_ratio=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert input_size % num_heads == 0
        self.input_size = input_size
        self.head_size = input_size // num_heads
        self.num_heads = num_heads
        self.attention_fn = attention_fn

        self.input_projections = nn.ModuleList(
            [nn.Linear(input_size, input_size) for _ in range(num_heads*3)])
        self.output_projection = nn.Linear(input_size, input_size)

        self.dropout_fn = nn.Dropout(p=dropout_ratio)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        head_dim = (batch_size, -1, self.num_heads, self.head_size)
        multi_head_dim = (batch_size, -1, self.num_heads * self.head_size)

        query, key, value = self.project_input(query, key, value, head_dim)
        attn, _ = self.attention_fn(query, key, value,
                                    dropout_fn=self.dropout_fn)
        output = self.project_output(attn, multi_head_dim)
        return output

    def project_input(self, query, key, value, dim):
        return (
            self.input_projections[0](query).view(*dim),
            self.input_projections[1](key).view(*dim),
            self.input_projections[2](value).view(*dim)
        )

    def project_output(self, data, dim):
        return self.output_projection(data.contiguous().view(*dim))


class Embeddings(nn.Module):
    """Token embedding layer"""

    def __init__(self, embedding_size, vocab_size):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.embedding_size)


class PositionalEncoding(nn.Module):
    """Positional encoding layer"""

    def __init__(self, embedding, dropout_ratio, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding = embedding
        self.embedding_size = embedding.embedding_size

        self.position_encoding = auto.Variable(torch.ones(max_len,
                                                          self.embedding_size))
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        # TODO: Implement this
        emb = self.embedding(x)
        # pos = self.position_encoding[:x.size(1), :]
        x = emb  # + pos
        return self.dropout(x)


def scaled_dot_attention(query, key, value, dropout_fn):
    scaling_factor = math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor
    p_attn = F.softmax(scores, dim=-1)
    p_attn = dropout_fn(p_attn)
    return torch.matmul(p_attn, value), p_attn


def create_encoder(vocab_size,
                   num_layers,
                   embedding_size,
                   encoding_size,
                   feed_forward_size,
                   num_attention_heads,
                   dropout_ratio,
                   initialize=True):
    # Haven't really thought this through
    assert embedding_size == encoding_size

    attn = MultiHeadedAttention(encoding_size,
                                num_attention_heads,
                                scaled_dot_attention,
                                dropout_ratio)
    ff = FeedForward(encoding_size, feed_forward_size, dropout_ratio)

    embedding = Embeddings(embedding_size, vocab_size)
    pos = PositionalEncoding(embedding, dropout_ratio)

    encoder_layer = EncoderLayer(encoding_size, attn, ff, dropout_ratio)
    encoder = Encoder(pos, encoder_layer, num_layers)

    if initialize:
        for p in encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return encoder
