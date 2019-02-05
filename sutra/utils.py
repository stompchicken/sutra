import logging
import time


def setup_logging():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)-20s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


class EarlyStopping(object):

    def __init__(self, patience=1):
        self.values = []
        self.patience = patience

    def update(self, metric):
        self.values.append(metric)

    def should_stop(self):
        if len(self.values) > self.patience + 1:
            v = list(reversed(self.values))
            return min(v[0:self.patience]) >= min(v[self.patience:])


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.duration = self.end - self.start
