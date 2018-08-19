from sutra.transformer import *


def test_scaled_dot_attention():
    attn = MultiHeadedAttention.scaled_dot_attention

    batch_size = 2
    num_heads = 3
    head_size = 2
    seq_length = 4
    dim = (2, 3, 2, 4)

    query = torch.zeros(dim)
    key = torch.zeros(dim)
    value = torch.zeros(dim)

    attn, potential = attn(query, key, value)
    assert attn.size() == dim


def test_encoder():
    encoder = create_encoder(vocab_size=100,
                             num_layers=2,
                             embedding_size=16,
                             encoding_size=16,
                             feed_forward_size=32,
                             num_attention_heads=4,
                             dropout_ratio=0.0)

    batch_size = 2
    seq_length = 5
    seq = torch.tensor([[0,1,2,3,4], [5,6,7,8,9]])
    encodings = encoder.forward(seq)
    assert encodings.size() == (batch_size, seq_length, 16)
