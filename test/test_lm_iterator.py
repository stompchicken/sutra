import sutra.data.data as data
import sutra.data.iterators as iterators

from test.asserts import assert_eq


def assert_lm_iterator(iterator, batch_data, batch_target):
    expected = [data.Batch(d, t) for d, t in zip(batch_data, batch_target)]
    batches = list(iterator)

    assert len(batches) == len(expected)

    for i in range(len(batches)):
        assert_eq(expected[i].data, batches[i].data)
        assert_eq(expected[i].target, batches[i].target)


def test_batched_lm_iterator():

    data = list(range(20))

    input = [
        [[0, 6, 12],
         [1, 7, 13]],
        [[2, 8, 14],
         [3, 9, 15]],
        [[4, 10, 16]]
    ]

    target = [
        [[1, 7, 13],
         [2, 8, 14]],
        [[3, 9, 15],
         [4, 10, 16]]
    ]

    assert_lm_iterator(iterators.BatchedLanguageModelIterator(
        data,
        batch_size=3,
        seq_length=2),
        input, target)

    input = [
        [[0, 6, 12],
         [1, 7, 13],
         [2, 8, 14]],
    ]

    target = [
        [[1, 7, 13],
         [2, 8, 14],
         [3, 9, 15]],
    ]

    assert_lm_iterator(iterators.BatchedLanguageModelIterator(
        data,
        batch_size=3,
        seq_length=3),
        input, target)

    input = [
        [[0, 10],
         [1, 11],
         [2, 12],
         [3, 13]],
        [[4, 14],
         [5, 15],
         [6, 16],
         [7, 17]]
    ]

    target = [
        [[1, 11],
         [2, 12],
         [3, 13],
         [4, 14]],
        [[5, 15],
         [6, 16],
         [7, 17],
         [8, 18]]
    ]

    assert_lm_iterator(iterators.BatchedLanguageModelIterator(
        data,
        batch_size=2,
        seq_length=4),
        input, target)


def test_lm_iterator():
    data = list(range(10))

    input = [
        [[0, 5],
         [1, 6],
         [2, 7]],
        [[1, 6],
         [2, 7],
         [3, 8]],
    ]

    target = [
        [3, 8],
        [4, 9],
    ]

    assert_lm_iterator(iterators.LanguageModelIterator(
        data,
        batch_size=2,
        seq_length=3),
        input, target)

    input = [
        [[0, 5],
         [1, 6]],
        [[1, 6],
         [2, 7]],
        [[2, 7],
         [3, 8]],
    ]

    target = [
        [2, 7],
        [3, 8],
        [4, 9],
    ]

    assert_lm_iterator(iterators.LanguageModelIterator(
        data,
        batch_size=2,
        seq_length=2),
        input, target)
