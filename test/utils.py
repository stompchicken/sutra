import torch

from sutra.data.data import Batch


def create_batch(text, target):
    return Batch(torch.LongTensor(text), torch.LongTensor(target))
