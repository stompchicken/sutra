import logging


def setup_logging():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)-20s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


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

    def get_estimate(self):
        return self.mean

    def __repr__(self):
        return '%s: %.5f' % (self.name, self.mean)


class EarlyStopping(object):

    def __init__(self, metric, patience=1):
        self.metric = metric
        self.values = []
        self.patience = patience

    def update(self, metrics):
        self.values.append(metrics[self.metric_name])

    def should_stop(self):
        if len(self.values) > self.patience + 1:
            v = list(reversed(self.values))
            return min(v[0:self.patience]) >= min(v[self.patience:])
