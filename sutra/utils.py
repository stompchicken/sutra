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

    def __init__(self, patience=1):
        self.values = []
        self.patience = patience

    def add_value(self, value):
        self.values.append(value)

    def should_stop(self):
        if len(self.values) > self.patience + 1:
            v = list(reversed(self.values))
            return min(v[0:self.patience]) >= min(v[self.patience:])
