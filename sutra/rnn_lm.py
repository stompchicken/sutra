import math
import copy

import torch
import torch.nn as nn
import torch.autograd as auto
import torch.optim as optim
import torch.nn.functional as F

from language_model import Dataset


class Metric(object):

    def __init__(self, name):
        self.name = name
        self.mean = 0.0
        self.updates = 1

    def update(self, metric):
        self.mean = self.mean + ((metric - self.mean) / self.updates)
        self.updates += 1

    def reset(self):
        self.mean = 0
        self.updates = 1

    def __repr__(self):
        return '%s: %.5f' % (self.name, self.mean)

class RNNEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, encoding_size, dropout_ratio):
        super(RNNEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.encoding_size = encoding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.num_layers = 1
        self.rnn = nn.LSTM(self.embedding_size, self.encoding_size, self.num_layers, dropout=dropout_ratio)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.encoding_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.encoding_size, device=device))

    def forward(self, input, hidden):
        embeddings = self.dropout(self.embedding(input))
        embeddings = embeddings.permute(1,0,2)
        output, hidden = self.rnn(embeddings, hidden)
        output = self.dropout(output)
        return output, hidden


class RNNLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, encoding_size):
        super(RNNLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size

        self.encoder = RNNEncoder(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  encoding_size=encoding_size,
                                  dropout_ratio=0.0)
        self.decoder = nn.Linear(encoding_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size, device):
        return self.encoder.init_hidden(batch_size, device)

    def forward(self, input, hidden):
        encoding, hidden = self.encoder.forward(input, hidden)
        decoded = self.decoder(encoding.view(encoding.size(0)*encoding.size(1), encoding.size(2)))
        return decoded.view(encoding.size(0), encoding.size(1), decoded.size(1)), hidden


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print("Device: %s" % device)

    vocab_size = 10000
    embedding_size = 64
    encoding_size = 64
    dataset = Dataset('data/language_modelling/wikitext-2/wiki.valid.tokens', vocab_size)
    model = RNNLanguageModel(vocab_size,
                             embedding_size=embedding_size,
                             encoding_size=encoding_size)
    model.to(device)

    batch_size = 32
    seq_length = 30
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    loss_estimate = Metric('loss')

    hidden = model.init_hidden(batch_size, device)


    print('Batch size: %d' % batch_size)
    print('Sequence length: %d' % seq_length)
    print('Corpus size: %d' % len(dataset.mapped_tokens))

    for epoch in range(10):
        it = dataset.batched_iterator(seq_length=seq_length, batch_size=batch_size)
        for i, batch in enumerate(it):

            current_batch_size = len(batch)
            
            context = torch.from_numpy(batch[:, :-1]).to(device)
            target = torch.from_numpy(batch[:, -1]).to(device)

            hidden = repackage_hidden(hidden)
#            hidden = (hidden[0][:, current_batch_size, :],
#                      hidden[1][:, current_batch_size, :])
#            
            optimizer.zero_grad()

            output, hidden = model(context, hidden)
            loss = criterion(output[-1], target)

            loss.backward()

            grad_clip = 5.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            loss_estimate.update(loss.item())

            optimizer.step()

            if i % 1000 == 0:
                print('[%d.%d] %s' % (epoch, i, loss_estimate))

        print('[%d.%d] %s' % (epoch, 0, loss_estimate))

        loss_estimate.reset()

if __name__ == '__main__':
    main()
