import torch
from numpy.testing import assert_almost_equal


def assert_eq(actual, expected, decimal=7):
    assert_almost_equal(to_numpy(actual), to_numpy(expected), decimal=decimal)


def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().numpy()
    else:
        return arr
