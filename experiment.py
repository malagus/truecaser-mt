import logging
import os
from typing import List

from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings
from flair.trainers import ModelTrainer

from category import Categories
from dataset.datasets import TruecaseDataset

log = logging.getLogger("truecaser")


class Experiment:
    def __init__(self,
                 categories: Categories,
                 embeddings: List[TokenEmbeddings],
                 corpus: Corpus,
                 tagger,
                 base_path: str,
                 max_epochs: int = 5,
                 mini_batch_size: int = 32
                 ):
        self.categories = categories
        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        # corpus._train = TruecaseDataset(corpus.train, categories=self.categories)
        # corpus._dev = TruecaseDataset(corpus.dev, categories=self.categories)
        # corpus._test = TruecaseDataset(corpus.test, categories=self.categories)
        self.corpus = corpus
        self.max_epochs = max_epochs
        self.mini_batch_size = mini_batch_size
        self.base_path = base_path
        self.tagger = tagger

    def train(self):
        checkpoint = self.base_path + '/checkpoint.pt'
        if os.path.isfile(checkpoint):
            log.info("Checkpoint exist resume training")
            trainer = ModelTrainer.load_checkpoint(checkpoint, self.corpus)
        else:
            trainer: ModelTrainer = ModelTrainer(self.tagger, self.corpus)

        trainer.train(self.base_path,
                      checkpoint=False,
                      train_with_dev=False,
                      max_epochs=self.max_epochs,
                      mini_batch_size=self.mini_batch_size,
                      embeddings_storage_mode='none')

    def eval(self):
        pass
