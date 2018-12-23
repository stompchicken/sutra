import math
import logging
import numpy as np

from sutra.data.data import Batch

logger = logging.getLogger(__name__)


class LanguageModelIterator:
    def __init__(self, dataset,
                 batch_size,
                 seq_length,
                 partial_batch=False,
                 pad_last_batch=False):
        """BPTT-style iterator
        :param partial_batch: Whether to include the final batch, if
        partial
        :param pad_last_batch: Whether to pad out final batch. This is
        for torchtext compatibility
        """

        data = np.array(dataset)

        if pad_last_batch:
            stride = (len(data) // batch_size) + 1
            pad_size = (stride * batch_size) - len(data)
            # TODO: Lookup padding index
            data = np.append(data, np.ones(pad_size, dtype=np.int64))
        else:
            stride = len(data) // batch_size

        # Split into `batch_size` separate columns, of length `stride`
        column_indices = []
        for i in range(batch_size):
            column_indices.append((i * stride, (i + 1) * stride))

        data = np.column_stack([data[i1:i2] for (i1, i2) in column_indices])

        self.data = data
        self.batch_size = batch_size
        self.stride = stride
        self.seq_length = seq_length
        self.num_batches = math.ceil(stride / seq_length)
        self.last_batch_is_partial = stride % seq_length != 0
        self.partial_batch = partial_batch

        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_batches:
            raise StopIteration
        elif self.index == self.num_batches - 1:
            if self.partial_batch:
                begin = self.seq_length * self.index
                # End index is one smaller so that all data has a target
                end = len(self.data) - 1
                batch = Batch(
                    self.data[begin:end],
                    self.data[begin + 1:end + 1])
                self.index += 1
                return batch
            else:
                raise StopIteration
        else:
            begin = self.seq_length * self.index
            end = self.seq_length * (self.index + 1)
            batch = Batch(
                self.data[begin:end],
                self.data[begin + 1:end + 1])
            self.index += 1
            return batch
