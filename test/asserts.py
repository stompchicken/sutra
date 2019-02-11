import numpy as np
from numpy.testing import assert_almost_equal

import torch


def assert_eq(actual, expected, decimal=6):
    assert_almost_equal(to_numpy(actual), to_numpy(expected), decimal=decimal)


def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().numpy()
    else:
        return arr


def assert_non_zero(data, decimal=6):
    return np.all(np.abs(to_numpy(data)) < 0.1 ** decimal)
