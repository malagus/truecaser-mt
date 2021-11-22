# Init wandb
import argparse
import distutils
import logging

import torch
import wandb
from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings, TokenEmbeddings, WordEmbeddings, ELMoEmbeddings, \
    StackedEmbeddings, FlairEmbeddings
# from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import Adam
from wandb.wandb_run import Run

from category import BaseWordCategories, Categories
from dataset.datasets import WIKINER_ENGLISH
from dataset.processor import DatasetProcessor
from model.sequence_tagger_model import SequenceTagger
from trainer import TruecaserTrainer

log = logging.getLogger("truecaser")


def get_corpus(corpus_name):
    if corpus_name == 'wikiner':
        return WIKINER_ENGLISH(base_path='data/base_word', tag_to_bioes=None, in_memory=True)
    elif corpus_name == 'conll03':
        corpus: Corpus = CONLL_03(base_path='data')
        processor = DatasetProcessor(use_char_tokenizer=False, categories=BaseWordCategories())
        corpus._train = processor.process(corpus.train)
        corpus._dev = processor.process(corpus.dev)
        corpus._test = processor.process(corpus.test)
        return corpus
    else:
        raise Exception("Unknown corpus")


def get_embeddings(embeddings_name, fine_tune=False):
    embeddings: TokenEmbeddings
    if embeddings_name == 'glove':
        embeddings = WordEmbeddings('glove')
    elif embeddings_name == 'extvec':
        embeddings = WordEmbeddings('extvec')
    elif embeddings_name == 'elmo':
        embeddings = ELMoEmbeddings()
    elif embeddings_name == 'flair':
        embeddings = StackedEmbeddings([
            FlairEmbeddings('news-forward'),
            FlairEmbeddings('news-backward'),
        ])
    else:
        embeddings = TransformerWordEmbeddings(embeddings_name, fine_tune=fine_tune)
    return embeddings


def main():
    """
    Base script for training rnn based model
    """
    # Training settings
    parser = argparse.ArgumentParser(description='Truecaser training')
    parser.add_argument('--corpus', type=str, default='wikiner')
    parser.add_argument('--embeddings', type=str, default='glove')
    # parser.add_argument('--fine_tune', type=bool, default=False)
    # CRF or softmax
    parser.add_argument('--use_crf', type=lambda x: bool(distutils.util.strtobool(x)), default='True')
    # RNN args
    parser.add_argument('--use_rnn', type=bool, default=True)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--rnn_type', type=str, default='LSTM')
    # Dropout
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--word_dropout', type=float, default=0.05)
    parser.add_argument('--locked_dropout', type=float, default=0.5)

    parser.add_argument('--mini_batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    args = parser.parse_args()
    run: Run = wandb.init(project="truecaser", group="rnn")

    wandb.config.update(args)

    categories: Categories = BaseWordCategories()
    corpus: Corpus = get_corpus(args.corpus)
    embeddings: TokenEmbeddings = get_embeddings(args.embeddings)

    model: SequenceTagger = SequenceTagger(
        embeddings=embeddings,
        tag_dictionary=categories,
        tag_type='case',
        hidden_size=args.hidden_size,
        use_rnn=args.use_rnn,
        rnn_layers=args.rnn_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        word_dropout=args.word_dropout,
        locked_dropout=args.locked_dropout,
        reproject_embeddings=False
    )
    wandb.watch(model)
    trainer: TruecaserTrainer = TruecaserTrainer(model, corpus, optimizer=Adam, use_wandb=True)
    base_path = 'result/' + run.id

    trainer.train(base_path=base_path,
                  mini_batch_size=args.mini_batch_size,
                  learning_rate=args.learning_rate,
                  max_epochs=args.max_epochs,
                  checkpoint=False,
                  train_with_dev=False,
                  # scheduler=OneCycleLR,
                  embeddings_storage_mode='none',
                  monitor_test=True,
                  monitor_train=True)


if __name__ == '__main__':
    main()
