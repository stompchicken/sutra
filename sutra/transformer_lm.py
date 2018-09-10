import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from language_model import Dataset
from transformer import create_encoder


class TransformerLanguageModel(nn.Module):

    def __init__(self, device, vocab_size):
        super(TransformerLanguageModel, self).__init__()

        self.encoder = create_encoder(vocab_size=vocab_size,
                                      num_layers=2,
                                      embedding_size=64,
                                      encoding_size=64,
                                      feed_forward_size=256,
                                      num_attention_heads=4,
                                      dropout_ratio=0.1)
        self.embeddings = self.encoder.embedding.embedding.embeddings.weight
        self.encoder.to(device)

    def forward(self, x):
        encodings = self.encoder.forward(x)[:, -1]
        return F.softmax(torch.matmul(encodings, torch.t(self.embeddings)), dim=1)


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


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)


    #dataset = Dataset('howl.txt', 500)
    vocab_size = 10000
    dataset = Dataset('data/language_modelling/wikitext-2/wiki.valid.tokens', vocab_size)
    model = TransformerLanguageModel(device, vocab_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    loss_estimate = Metric('loss')

    for epoch in range(3):
        for i, batch in enumerate(dataset.batched_iterator(seq_length=50, batch_size=128)):
            context = torch.from_numpy(batch[:, :-2]).to(device)
            token = torch.from_numpy(batch[:, -1]).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(context)
            loss = criterion(output, token)

            loss_estimate.update(loss.item())
            loss.backward()
            optimizer.step()


        print('[%d] %s' % (epoch, loss_estimate))

        loss_estimate.reset()

if __name__ == '__main__':
    main()
