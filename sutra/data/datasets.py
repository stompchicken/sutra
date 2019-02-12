import io
import pathlib
import requests
import zipfile
import logging

from sutra.data.data import Corpus, Document

logger = logging.getLogger(__name__)


def get_dataset(name, vocab_size, cache=None):
    """Utility method to get a dataset by name lookup"""

    cache = cache or DatasetCache()

    return {
        'wikitext-2': WikiText2(cache, vocab_size=vocab_size)
    }[name]


class DatasetCache:
    def __init__(self):
        self.cache_dir = '.data_cache'

    def dataset_is_cached(self, dataset_name):
        return pathlib.Path(self.cache_dir, dataset_name).exists()


class WikiText2:
    """Wikitext-2 language modelling dataset"""

    name = 'wikitext-2'
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'

    @classmethod
    def download(cls, cache):
        logging.info('Downloading {}'.format(cls.name))
        req = requests.get(cls.url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(req.content))
        z.extractall(cache.cache_dir)
        logging.debug('Extracted {} to {}'.format(cls.name, cache.cache_dir))

    def __init__(self, cache, vocab_size=None, lowercase=False, specials=None):
        """
        Creates the dataset, potentially downloading if not present in the cache

        Args:
            cache: DatasetCache
            vocab_size: Maximum size of the vocabulary to generate
            lowercase: Convert all tokens to lower case
            specials: List of reserved tokens to put in the vocab
        """

        path = pathlib.Path(cache.cache_dir, self.name)

        if not cache.dataset_is_cached(self.name):
            self.download(cache)

        train_tokens = self.__load_from_tokens_file(
            pathlib.Path(path, 'wiki.train.tokens'), lowercase)
        valid_tokens = self.__load_from_tokens_file(
            pathlib.Path(path, 'wiki.valid.tokens'), lowercase)
        test_tokens = self.__load_from_tokens_file(
            pathlib.Path(path, 'wiki.test.tokens'), lowercase)

        # Create vocab
        train_corpus = Corpus([Document(train_tokens)])
        specials = specials or []
        self.vocab = train_corpus.create_vocab(
            vocab_size, specials=specials)

        # Create vocab-mapped data
        self.train_data = [
            self.vocab.term_to_index(token)
            for token in train_tokens
        ]

        self.valid_data = [
            self.vocab.term_to_index(token)
            for token in valid_tokens
        ]

        self.test_data = [
            self.vocab.term_to_index(token)
            for token in test_tokens
        ]

    def __load_from_tokens_file(self, path, lowercase=False):
        """Load tokens from a space-separated text file.

        Special <eos> tokens are added at the end of each line, which
        is the standard for the Wikitext data

        Returns:
           list: A list of tokens
        """

        tokens = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                if lowercase:
                    tokens.extend(token.lower() for token in line.split())
                else:
                    tokens.extend(line.split())

                tokens.append('<eos>')
        return tokens
