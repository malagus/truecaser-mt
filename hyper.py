from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter, OptimizationValue, SequenceTaggerParamSelector

from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, \
    CharacterEmbeddings, BertEmbeddings, FlairEmbeddings, ELMoEmbeddings, TransformerWordEmbeddings
from typing import List
from datetime import datetime
# 1. get the corpus
from category import Categories, BaseWordCategories, BaseCharCategories
from data import TruecaserSentence, TruecaserCorpus
from dataset.datasets import Conll_2003_Trucase, Wiki_Data, TruecaseDataset, WIKINER_ENGLISH

# corpus: Corpus = Conll_2003_Trucase(base_path='data')
from dataset.processor import DatasetProcessor, character_tokenizer
from model.rnn.char import CharRNNTrueceser

# corpus: Corpus = CONLL_03(base_path='data')
#
# processor = DatasetProcessor(use_char_tokenizer=False, categories=BaseWordCategories())
# corpus._train = processor.process(corpus.train)
# corpus._dev = processor.process(corpus.dev)
# corpus._test = processor.process(corpus.test)

corpus: Corpus = WIKINER_ENGLISH(base_path='data/base_word', tag_to_bioes=None, in_memory=True).downsample()
# define your search space
search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    WordEmbeddings('glove'),
    ELMoEmbeddings(),
    StackedEmbeddings([FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')])
])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])
categories: Categories = BaseWordCategories()
#
#
# # 3. make the tag dictionary from the corpus
# # tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
tag_dictionary = categories
# # initialize embeddings
# embedding_types: List[TokenEmbeddings] = [
#
#     # GloVe embeddings
#     # WordEmbeddings('glove'),
#     ELMoEmbeddings(),
#     # FlairEmbeddings('news-forward'),
#     # FlairEmbeddings('news-backward'),
#     # BertEmbeddings('bert-base-uncased'),
#     # CharacterEmbeddings(),
#
#     # # contextual string embeddings, forward
#     # PooledFlairEmbeddings('news-forward', pooling='min'),
#     #
#     # # contextual string embeddings, backward
#     # PooledFlairEmbeddings('news-backward', pooling='min'),
# ]
# now = datetime.now()
# # print(now)
# # embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
# embeddings = TransformerWordEmbeddings('bert-base-uncased')
#
# #
# # # initialize sequence tagger
# from flair.models import SequenceTagger
#
# tagger: SequenceTagger = SequenceTagger(  # hidden_size=256,512
#     hidden_size=256,
#     rnn_layers=2,
#     rnn_type='LSTM',
#     embeddings=embeddings,
#     tag_dictionary=tag_dictionary,
#     tag_type='case',
#     use_rnn=False
# )
#
# from experiment import Experiment
#
# experiment: Experiment = Experiment(categories=categories,
#                                     embeddings=embedding_types,
#                                     corpus=corpus,
#                                     tagger=tagger,
#                                     base_path='result/wiki_ner/base_word/bert-base-uncased-CRF'
#                                     )
# experiment.train()

param_selector = SequenceTaggerParamSelector(
    corpus=corpus,
    tag_type='case',
    base_path='result/hyper'

)

param_selector.optimize(search_space, max_evals=100)
