import os
import tempfile

from sutra.language_model import *
from test.asserts import assert_eq

def test_dataset():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, 'test_corpus.txt')
    with open(path, 'w', encoding='utf8') as f:
        f.write("This is a test sentence is a test")

    dataset = Dataset(path, vocab_size=4)

    assert dataset.vocab.term_to_index('This') == 0
    assert dataset.vocab.term_to_index('is') == 1
    assert dataset.vocab.term_to_index('a') == 2
    assert dataset.vocab.term_to_index('test') == 3
    assert dataset.vocab.term_to_index('sentence') == 0
    assert dataset.vocab.term_to_index('OOV') == 0

    assert_eq(dataset.mapped_tokens, np.array([0, 1, 2, 3, 0, 1, 2, 3]))

    batches = dataset.batched_iterator(3, 6)
    assert_eq(batches[0], [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2], [1, 2, 3]])

    batches = dataset.batched_iterator(3, 2)
    assert_eq(batches[0], [[0, 1, 2], [1, 2, 3]])
    assert_eq(batches[1], [[2, 3, 0], [3, 0, 1]])
    assert_eq(batches[2], [[0, 1, 2], [1, 2, 3]])

    batches = dataset.batched_iterator(4, 3)
    assert_eq(batches[0], [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]])
    assert_eq(batches[1], [[3, 0, 1, 2], [0, 1, 2, 3]])
