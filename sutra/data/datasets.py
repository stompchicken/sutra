import io
import pathlib
import requests
import zipfile
import logging

from sutra.data.data import Corpus, Document

logger = logging.getLogger(__name__)


class DatasetCache:
    def __init__(self):
        self.cache_dir = '.data_cache'

    def dataset_is_cached(self, dataset_name):
        return pathlib.Path(self.cache_dir, dataset_name).exists()


class WikiText2:
    name = 'wikitext-2'
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'

    @classmethod
    def download(cls, cache):
        logging.info('Downloading {}'.format(cls.name))
        req = requests.get(cls.url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(req.content))
        z.extractall(cache.cache_dir)
        logging.debug('Extracted {} to {}'.format(cls.name, cache.cache_dir))

    def __init__(self, cache, vocab_size=None, lowercase=False):
        path = pathlib.Path(cache.cache_dir, self.name)

        if not cache.dataset_is_cached(self.name):
            self.download(cache)

        self.train_tokens = self.__load_from_tokens_file(
            pathlib.Path(path, 'wiki.train.tokens'), lowercase)
        self.valid_tokens = self.__load_from_tokens_file(
            pathlib.Path(path, 'wiki.valid.tokens'), lowercase)
        self.test_tokens = self.__load_from_tokens_file(
            pathlib.Path(path, 'wiki.test.tokens'), lowercase)

        self.train_corpus = Corpus([Document(self.train_tokens)])
        self.vocab = self.train_corpus.create_vocab(
            vocab_size, specials=['<unk>', '<pad>'])

        self.valid_corpus = Corpus([Document(self.valid_tokens)])
        self.test_corpus = Corpus([Document(self.test_tokens)])

        self.train_data = [
            self.vocab.term_to_index(token)
            for token in self.train_tokens
        ]

        self.valid_data = [
            self.vocab.term_to_index(token)
            for token in self.valid_tokens
        ]

        self.test_data = [
            self.vocab.term_to_index(token)
            for token in self.test_tokens
        ]

    def __load_from_tokens_file(self, path, lowercase=False):
        tokens = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                if lowercase:
                    tokens.extend(token.lower() for token in line.split())
                else:
                    tokens.extend(line.split())

                tokens.append('<eos>')
        return tokens
