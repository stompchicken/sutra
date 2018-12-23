import collections
import itertools
import json

import pandas as pd

TrainingConfig = collections.namedtuple('TrainingConfig', [
    'epoch_length',
    'max_epochs',
    'optimizer',
])

class Metrics:

    def __init__(self):
        self.data = pd.DataFrame()

    def append(self, metrics):
        self.data = self.data.append(metrics)


class Trainer:

    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.optimizer = config.optimizer
        self.epoch_length = config.epoch_length

        self.metrics = Metrics()

    def epoch(self, iterations):
        return iteration / self.epoch_length

    def end_of_epoch(self, iteration):
        return (iteration + 1) % self.epoch_length == 0

    def train(self, train_dataset, eval_dataset):

        self.model.train()

        for i, batch in enumerate(train_dataset):
            start = time.time()

            optimizer.zero_grad()
            metrics = self.model(batch)
            loss = metrics['loss']
            loss.backward()
            optimizer.step()

            end = time.time()

            metrics['stage'] = 'train'
            metrics['iteration'] = i
            metrics['duration'] = end - start
            self.metrics.append(metrics)

            if self.end_of_epoch(i) == 0:
                self.evaluate(eval_dataset)
                torch.save(self.model.state_dict(), 'rnn_lm.model')
                self.model.train()

    def evaluate(self, eval_dataset):
        model.evaluate()
        for i, batch in enumerate(eval_dataset):
            start = time.time()
            metrics = model(batch)
            end = time.time()

            metrics['stage'] = 'evaluate'
            metrics['iteration'] = i
            metrics['duration'] = end - start

            self.metrics.append(metrics)
