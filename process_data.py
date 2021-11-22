from pathlib import Path

from flair.data import Corpus
from flair.datasets import CONLL_03, WIKINER_ENGLISH

from category import BaseWordCategories
from dataset.datasets import TruecaseDataset
# Preprocessing script for datasets
base_path = 'data/base_word'
base_path: Path = Path(base_path)
corpus: Corpus = WIKINER_ENGLISH(base_path='data', tag_to_bioes=None)

# column format
columns = {0: "text", 1: "pos", 2: "ner"}

# this dataset name
dataset_name = corpus.__class__.__name__.lower()

data_folder = base_path / dataset_name

corpus._train = TruecaseDataset(corpus.train, categories=BaseWordCategories())
corpus._dev = TruecaseDataset(corpus.dev, categories=BaseWordCategories())
corpus._test = TruecaseDataset(corpus.test, categories=BaseWordCategories())


def save_dataset(location, dataset):
    with open(data_folder / location, 'w+') as f:
        for sentence in dataset:
            for token in sentence:
                f.write(
                    f"{token.text}\t{token.get_tag('pos').value}\t{token.get_tag('ner').value}\t{token.get_tag('case').value}")
                f.write("\n")
            f.write("\n")


save_dataset("aij-wikiner-en-wp3.train", corpus.train)
save_dataset("aij-wikiner-en-wp3.dev", corpus.dev)
save_dataset("aij-wikiner-en-wp3.test", corpus.test)
