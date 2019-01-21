import logging
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from sutra.transformer import create_encoder


logger = logging.getLogger(__name__)


class TransformerLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, encoding_size, device):
        super(TransformerLanguageModel, self).__init__()

        self.encoder = create_encoder(vocab_size=vocab_size,
                                      num_layers=1,
                                      embedding_size=embedding_size,
                                      encoding_size=encoding_size,
                                      feed_forward_size=256,
                                      num_attention_heads=4,
                                      dropout_ratio=0.1)
        self.encoder.to(device)

    def forward(self, x):
        encodings = self.encoder.forward(x)[:, -1]
        # Tied embeddings
        embeddings = self.encoder.embedding.embedding.embeddings.weight
        return F.softmax(torch.matmul(encodings, torch.t(embeddings)), dim=1)


TransformerLanguageModelConfig = collections.namedtuple(
    'TransformerLanguageModelConfig',
    ['vocab_size', 'seq_length', 'batch_size',
     'embedding_size', 'encoding_size'])


# def main():
#     utils.setup_logging()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     logging.info("Device: %s" % device)

#     config = TransformerLanguageModelConfig(
#         vocab_size=30000,
#         batch_size=2,
#         seq_length=5,
#         embedding_size=4,
#         encoding_size=4)

#     train, valid, test = lm.load_wikitext2(config.vocab_size)

#     train_iter, valid_iter, test_iter = lm.iterator(train, valid, test,
#                                      config.batch_size, device)

#     model = TransformerLanguageModel(config.vocab_size,
#                                      config.embedding_size,
#                                      config.encoding_size,
#                                      device)

#     model.train()

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
#                                                      patience=4,
#                                                      verbose=True)
#     early_stopping = utils.EarlyStopping(6)


#     loss_estimate = utils.Metric('loss')
#     perplexity = utils.Metric('ppl')
#     tokens_per_second = utils.Metric('tokens/s')


#     for batch in train_iter:

#         import pdb; pdb.set_trace()

#         optimizer.zero_grad()

#         output = model(batch.text)

#         optimizer.step()


#         break


# if __name__ == '__main__':
#    main()
