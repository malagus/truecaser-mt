# Init wandb
import argparse
import distutils
import logging
import os

import torch
import wandb
from flair.data import Corpus
from flair.datasets import CONLL_03, WIKINER_ENGLISH
from flair.embeddings import TransformerWordEmbeddings, TokenEmbeddings, WordEmbeddings

from torch.optim import Adam
from wandb.wandb_run import Run

from category import BaseWordCategories, Categories
from dataset.datasets import WIKINER_ENGLISH
from dataset.processor import DatasetProcessor
from model.sequence_tagger_model import SequenceTagger
from optim import STLR
from trainer import TruecaserTrainer

log = logging.getLogger("truecaser")


def get_corpus(corpus_name):
    if corpus_name == 'wikiner':
        return WIKINER_ENGLISH(base_path='data/base_word', tag_to_bioes=None, in_memory=True)
        # corpus: Corpus = WIKINER_ENGLISH(in_memory=True)
        # processor = DatasetProcessor(use_char_tokenizer=False, categories=BaseWordCategories())
        # corpus._train = processor.process(corpus.train)
        # corpus._dev = processor.process(corpus.dev)
        # corpus._test = processor.process(corpus.test)
        # return corpus
    elif corpus_name == 'conll03':
        corpus: Corpus = CONLL_03(base_path='data')
        processor = DatasetProcessor(use_char_tokenizer=False, categories=BaseWordCategories())
        corpus._train = processor.process(corpus.train)
        corpus._dev = processor.process(corpus.dev)
        corpus._test = processor.process(corpus.test)
        return corpus
    else:
        raise Exception("Unknown corpus")


def get_embeddings(embeddings_name):
    return TransformerWordEmbeddings(embeddings_name, fine_tune=True)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Truecaser training')
    parser.add_argument('--corpus', type=str, default='wikiner')
    parser.add_argument('--embeddings', type=str, default='glove')
    # CRF or softmax
    parser.add_argument('--use_crf', type=lambda x:bool(distutils.util.strtobool(x)), default='True')
    # Dropout
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--word_dropout', type=float, default=0.05)
    parser.add_argument('--locked_dropout', type=float, default=0.5)

    parser.add_argument('--mini_batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    args = parser.parse_args()
    run: Run = wandb.init(project="truecaser", group="transformers_fine_tuning")

    wandb.config.update(args)
    wandb.config.fine_tune = True

    categories: Categories = BaseWordCategories()
    corpus: Corpus = get_corpus(args.corpus)
    embeddings: TokenEmbeddings = get_embeddings(args.embeddings)

    model: SequenceTagger = SequenceTagger(
        embeddings=embeddings,
        tag_dictionary=categories,
        tag_type='case',
        hidden_size=32,
        use_rnn=False,
        dropout=args.dropout,
        word_dropout=args.word_dropout,
        locked_dropout=args.locked_dropout,
        reproject_embeddings=False
    )
    wandb.watch(model)
    trainer: TruecaserTrainer = TruecaserTrainer(model, corpus, optimizer=Adam, use_wandb=True)
    base_path = wandb.run.dir
    log.info(args)

    trainer.train(base_path=base_path,
                  mini_batch_size=args.mini_batch_size,
                  learning_rate=args.learning_rate,
                  max_epochs=args.max_epochs,
                  checkpoint=False,
                  train_with_dev=False,
                  scheduler=STLR,
                  embeddings_storage_mode='none',
                  monitor_test=True,
                  monitor_train=True)
    # wandb.save(os.path.join(wandb.run.dir, "final-model*"))
    # wandb.save(os.path.join(wandb.run.dir, "best-model*"))
    wandb.save(os.path.join(wandb.run.dir, "*"))


if __name__ == '__main__':
    main()
