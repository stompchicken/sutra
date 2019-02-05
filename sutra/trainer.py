import time
import logging
import typing
import math
import os
import json

import pandas as pd
import torch

from sutra.utils import EarlyStopping

logger = logging.getLogger(__name__)


class Experiment:

    def __init__(self):
        self.experiment_id = 0
        for i in range(1000):
            if not os.path.exists(f"exp/{i}"):
                self.experiment_id = i
                break

        logger.info(f"Starting exeriment id={self.experiment_id}")
        os.makedirs(f'exp/{self.experiment_id}')
        self.path = f'exp/{self.experiment_id}'


class TrainingConfig(typing.NamedTuple):
    epoch_length: int
    max_epochs: int
    batch_size: int
    optimizer: str


class Metrics:

    def __init__(self):
        self.data = pd.DataFrame()

    def append(self, metrics):
        self.data = self.data.append(metrics, ignore_index=True)

    def save(self, path):
        self.data.to_parquet(os.path.join(path, "metrics.parquet"),
                             compression=None)


class Trainer:

    def __init__(self, config, model, train_fn, eval_fn, optimizer):
        self.experiment = Experiment()
        self.config = config
        self.model = model
        self.train_fn = train_fn
        self.eval_fn = eval_fn

        self.optimizer = optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                       patience=3)
        self.early_stopping = EarlyStopping(5)

        self.epoch_length = config.epoch_length

        self.metrics = Metrics()

        config = {
            "training_config": config._asdict(),
            "model": model.config()
        }
        with open(os.path.join(self.experiment.path, "config"), 'w', encoding='utf8') as f:
            f.write(json.dumps(config, indent=2))

    def epoch(self, iteration):
        return iteration // self.epoch_length

    def end_of_epoch(self, iteration):
        return (iteration + 1) % self.epoch_length == 0

    def log_metrics(self, i, epoch, stage, duration, output):
        metrics = {}
        metrics['examples'] = self.config.batch_size
        metrics['epoch'] = epoch
        metrics['stage'] = stage
        metrics['iteration'] = i
        metrics['duration'] = duration
        metrics['loss'] = output['loss'].item()
        metrics['perplexity'] = math.exp(metrics['loss'])
        self.metrics.append(metrics)

    def print_training_metrics(self, i):
        if i % 100 == 0 and i > 0:
            current_epoch = self.epoch(i)
            df = self.metrics.data

            df = df[df.stage == 'train']
            df = df[df.iteration > i - 100]
            if len(df) > 0:
                examples_per_second = sum(df.examples) / sum(df.duration)
                avg_loss = sum(df.loss) / len(df)
                logging.info(f"[train] i={i} epoch={current_epoch} ex/s={examples_per_second:.0f} loss={avg_loss:.4f}") # noqa
            else:
                logging.warning(f"No data for training epoch {current_epoch}")

    def print_eval_metrics(self, epoch):
        df = self.metrics.data
        df = df[df.epoch == epoch]
        df = df[df.stage == 'evaluate']
        if len(df) > 0:
            avg_loss = sum(df.loss) / len(df)
            avg_ppl = sum(df.perplexity) / len(df)
            logging.info(f"[eval] epoch={epoch}, loss={avg_loss:.4f}, ppl={avg_ppl:.4f}") # noqa
        else:
            logging.warning(f"No data for evaluation epoch {epoch}")

    def train(self, train_dataset, eval_dataset):
        logger.info(self.config)

        model_state = self.model.init_state(self.config.batch_size)

        for i, batch in enumerate(train_dataset):
            if self.epoch(i) > self.config.max_epochs:
                logging.info("Reached maximum epochs")
                break

            if self.early_stopping.should_stop():
                logging.info("Reached early stopping condition")
                break

            start = time.time()

            self.optimizer.zero_grad()
            self.model.train()
            output, model_state = self.train_fn(batch, model_state)
            loss = output['loss']
            loss.backward()

            grad_clip = 5.0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()
            end = time.time()

            epoch = self.epoch(i)
            self.log_metrics(i, epoch, 'train', end - start, output)
            self.print_training_metrics(i)

            if self.end_of_epoch(i):
                self.evaluate(eval_dataset, self.epoch(i))
                self.model.train()

    def evaluate(self, eval_dataset, epoch):
        self.model.eval()

        model_state = self.model.init_state(self.config.batch_size)
        for i, batch in enumerate(eval_dataset):
            start = time.time()
            output, model_state = self.eval_fn(batch, model_state)
            end = time.time()

            self.log_metrics(i, epoch, 'evaluate', end - start, output)

        self.print_eval_metrics(epoch)

        self.checkpoint()

        df = self.metrics.data
        df = df[df.epoch == epoch]
        df = df[df.stage == 'evaluate']
        if len(df) > 0:
            avg_loss = sum(df.loss) / len(df)

        self.lr_scheduler.step(avg_loss)

        self.early_stopping.update(avg_loss)

    def checkpoint(self):
        self.metrics.save(self.experiment.path)
        torch.save(self.model.state_dict(), os.path.join(self.experiment.path, 'model'))
