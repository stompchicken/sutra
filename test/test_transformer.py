import torch
from test.asserts import assert_eq

from sutra.model.transformer import scaled_dot_attention
from sutra.model.transformer import MultiHeadedAttention, create_encoder


NO_DROPOUT = torch.nn.Dropout(0.0)
ALL_DROPOUT = torch.nn.Dropout(1.0)


def test_scaled_dot_attention_dropout():
    dim = (5, 5, 5, 5)

    query = torch.rand(dim)
    key = torch.rand(dim)
    value = torch.rand(dim)
    attn, potential = scaled_dot_attention(query, key, value, ALL_DROPOUT)

    # Droput 1.0 => everything is zero
    assert_eq(potential, torch.zeros(dim))
    assert_eq(attn, torch.zeros(dim))


def test_scaled_dot_attention_potentials():
    dim = (5, 5, 5, 5)

    query = torch.rand(dim)
    key = torch.rand(dim)
    value = torch.rand(dim)
    output, potential = scaled_dot_attention(query, key, value, NO_DROPOUT)

    # Potentials are multiplied by value to get output
    assert_eq(output, torch.matmul(potential, value))

    # Attention potential distribution columns sum to one
    assert_eq(torch.sum(potential, dim=-1), torch.ones((5, 5, 5)))


def test_scaled_dot_attention():
    dim = (10, 5, 4, 4)
    query = torch.rand(dim)
    key = torch.rand(dim)
    value = torch.rand(dim)

    output, potential = scaled_dot_attention(query, key, value, NO_DROPOUT)

    # Output, potentials and input should all be the same size
    assert output.size() == potential.size() == dim


def test_multi_headed_attention():

    # Plug in simple attention function
    def attn_fn(query, key, value, dropout_fn=None):
        return query + key + value, query

    input_size = 3
    batch_size = 2
    seq_length = 4
    dim = (batch_size, seq_length, input_size)

    multi_head_attn = MultiHeadedAttention(input_size=input_size,
                                           num_heads=1,
                                           attention_fn=attn_fn,
                                           dropout_prob=0.0)

    query = torch.rand(*dim)
    key = torch.rand(*dim)
    value = torch.rand(*dim)

    encodings = multi_head_attn.forward(query, key, value)

    assert encodings.size() == dim

    # Apply input projections, attention and output projection
    query_proj = multi_head_attn.input_projections[0](query)
    key_proj = multi_head_attn.input_projections[1](key)
    value_proj = multi_head_attn.input_projections[2](value)
    expected, _ = attn_fn(query_proj, key_proj, value_proj)
    expected = multi_head_attn.output_projection(expected)

    assert_eq(encodings, expected)


def test_encoder():
    encoder = create_encoder(vocab_size=100,
                             num_layers=2,
                             embedding_size=16,
                             encoding_size=16,
                             feed_forward_size=32,
                             num_attention_heads=4,
                             dropout_prob=0.0)

    batch_size = 2
    seq_length = 5
    seq = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    encodings = encoder.forward(seq)
    assert encodings.size() == (batch_size, seq_length, 16)
