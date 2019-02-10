import logging

import torch
import numpy as np

from sutra.data.data import Batch

logger = logging.getLogger(__name__)


def split_sequence_data(data, batch_size, pad_to_batch_size=False):
    """Split sequence data into a tensor with batch_size columns"""

    data = np.array(data)

    if pad_to_batch_size:
        stride = (len(data) // batch_size) + 1
        pad_size = (stride * batch_size) - len(data)
        # TODO: Lookup padding index rather than use ones
        data = np.append(data, np.ones(pad_size, dtype=np.int64))
    else:
        stride = len(data) // batch_size

    column_indices = []
    for i in range(batch_size):
        column_indices.append((i * stride, (i + 1) * stride))
    data = np.column_stack([data[i1:i2] for (i1, i2) in column_indices])

    data = torch.LongTensor(data)
    return data


class LanguageModelIterator:

    def __init__(self,
                 dataset,
                 batch_size,
                 seq_length,
                 pad_to_batch_size=False):

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.pad_to_batch_size = pad_to_batch_size

        self.data = split_sequence_data(dataset, batch_size, pad_to_batch_size)
        self.stride = self.data.size(0)
        # We don't fuck with partial batches
        self.num_full_batches = self.stride - self.seq_length
        logger.info(f"{self.num_full_batches} batches in data")
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):

        if self.index < self.num_full_batches:
            begin = self.index
            end = begin + self.seq_length

            # End index is one smaller so that all data has a target
            end = min(end, self.stride - 1)

            batch = Batch(
                self.data[begin:end],
                self.data[end])
        else:
            raise StopIteration

        self.index += 1

        return batch


# TODO: Make this a pytorch dataset
class BatchedLanguageModelIterator:
    def __init__(self, dataset,
                 batch_size,
                 seq_length,
                 pad_to_batch_size=False):
        """BPTT-style iterator for language model training and evaluation
        Args:
            allow_partial_batches: Whether to include the final batch, if it is not batch_size
            pad_to_batch_size: Whether to pad out data to batch size.
                               (This is for torchtext compatibility)
        """
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.pad_to_batch_size = pad_to_batch_size

        self.data = split_sequence_data(dataset, batch_size, pad_to_batch_size)
        self.stride = self.data.size(0)
        self.num_full_batches = (self.stride - self.seq_length) / self.seq_length
        logger.info(f"{self.num_full_batches} batches in data")

        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):

        if self.index < self.num_full_batches:
            begin = self.seq_length * self.index
            end = begin + self.seq_length

            # End index is one smaller so that all data has a target
            end = min(end, self.stride - 1)

            batch = Batch(
                self.data[begin:end],
                self.data[begin + 1:end + 1])
        else:
            raise StopIteration

        self.index += 1

        return batch
