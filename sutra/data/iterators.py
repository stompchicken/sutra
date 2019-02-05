import math
import logging

import torch
import numpy as np

from sutra.data.data import Batch

logger = logging.getLogger(__name__)


class LanguageModelIterator:
    def __init__(self, dataset,
                 batch_size,
                 seq_length,
                 device='cpu',
                 allow_partial_batch=False,
                 pad_last_batch=False,
                 repeat=False):
        """BPTT-style iterator for language model training and evaluation
        Args:
            partial_batch: Whether to include the final batch, if it is not batch_size
            pad_last_batch: Whether to pad out final batch. (This is for torchtext compatibility)
            repeate: Create 'infinite' iterators that loop forever
        """
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device
        self.pad_last_batch = pad_last_batch
        self.allow_partial_batch = allow_partial_batch
        self.repeat = repeat

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
        data = torch.LongTensor(data)

        self.data = data
        self.stride = stride
        self.num_batches = math.ceil(stride / seq_length)
        self.last_batch_is_partial = stride % seq_length != 0

        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):

        out_of_bounds = self.index >= self.num_batches
        final_batch = self.index == self.num_batches - 1

        if out_of_bounds:
            raise StopIteration
        elif final_batch:
            if self.allow_partial_batch:
                begin = self.seq_length * self.index
                # End index is one smaller so that all data has a target
                end = len(self.data) - 1
                batch = Batch(
                    self.data[begin:end],
                    self.data[begin + 1:end + 1])
            else:
                raise StopIteration
        else:
            begin = self.seq_length * self.index
            end = self.seq_length * (self.index + 1)
            batch = Batch(
                self.data[begin:end],
                self.data[begin + 1:end + 1])

        self.index += 1

        if self.repeat and not self.__valid_batch(self.index):
            self.index = 0

        return batch

    def __valid_batch(self, i):
        if self.index >= self.num_batches:
            return False
        elif self.index == self.num_batches - 1:
            if not self.last_batch_is_partial:
                return True
            elif self.allow_partial_batch:
                return True
            else:
                return False
        else:
            return True
