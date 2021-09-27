# from flair.data import Corpus, Sentence, Token
# from flair.datasets import CONLL_03
# from flair.models import SequenceTagger
# from pytest import collect
#
# from flair.data import Corpus
# from flair.datasets import CONLL_03
# from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, \
#     CharacterEmbeddings, BertEmbeddings, FlairEmbeddings
# from typing import List
# from datetime import datetime
# # 1. get the corpus
# from category import Categories, BaseWordCategories, BaseCharCategories
# from data import TruecaserSentence, TruecaserCorpus
# from dataset.datasets import Conll_2003_Trucase, Wiki_Data, TruecaseDataset, WIKINER_ENGLISH
#
# # corpus: Corpus = Conll_2003_Trucase(base_path='data')
# from dataset.processor import DatasetProcessor, character_tokenizer
# from experiment import Experiment
# from model.rnn.char import CharRNNTrueceser
#
# corpus: Corpus = WIKINER_ENGLISH(base_path='data', tag_to_bioes=None)
# # corpus: Corpus = CONLL_03(base_path='data')
# # corpus = corpus.downsample(0.01)
# print(corpus.train)
# processor = DatasetProcessor(use_char_tokenizer=True, categories=BaseCharCategories())
# # corpus._train = processor.process(corpus.train)
# # corpus._dev = processor.process(corpus.dev)
# # corpus._test = processor.process(corpus.test)
# corpus._train = TruecaseDataset(corpus.train)
# corpus._dev = TruecaseDataset(corpus.dev)
# corpus._test = TruecaseDataset(corpus.test)
# # corpus: Corpus = Wiki_Data(base_path='data')
# # 2. what tag do we want to predict?
# tag_type = 'case'
#
# categories: Categories = BaseCharCategories()
# #
# #
# # # 3. make the tag dictionary from the corpus
# # # tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
# tag_dictionary = categories
# # # initialize embeddings
# embedding_types: List[TokenEmbeddings] = [
#
#     # GloVe embeddings
#     # WordEmbeddings('glove'),
#     # FlairEmbeddings('news-forward'),
#     # FlairEmbeddings('news-backward'),
#     # BertEmbeddings(),
#     CharacterEmbeddings(),
#
#     # # contextual string embeddings, forward
#     # PooledFlairEmbeddings('news-forward', pooling='min'),
#     #
#     # # contextual string embeddings, backward
#     # PooledFlairEmbeddings('news-backward', pooling='min'),
# ]
# now = datetime.now()
# # print(now)
# embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
# #
# # # initialize sequence tagger
# # from flair.models import SequenceTagger
# #
# # tagger: SequenceTagger = SequenceTagger(  # hidden_size=256,512
# #     hidden_size=256,
# #     rnn_layers=2,
# #     embeddings=embeddings,
# #     tag_dictionary=tag_dictionary,
# #     tag_type=tag_type)
# tagger = CharRNNTrueceser(
#     rnn_hidden_size=256,
#     rnn_layers=2,
#
# )
# # # initialize trainer
# from flair.trainers import ModelTrainer
#
# trainer: ModelTrainer = ModelTrainer(tagger, corpus)
# experiment = Experiment()
# trainer.train('result/truecase/' + str(now),
#               train_with_dev=True,
#               max_epochs=10, mini_batch_size=32)
# # trainer.train('result/truecase/' + 'test',
# #               train_with_dev=True,
# #               max_epochs=3, mini_batch_size=32)
#
# test = Sentence("i love warsaw", use_tokenizer=character_tokenizer)
# tagger.predict(test)
# for token in test:
#     print(f"{token.text}\t{token.get_tag('case').value}")
#
# # test = TruecaserSentence(test)
# # for token in test:
# #     print(f"{token.text}\t{token.get_tag('case').value}")
# # embeddings.embed(test)
#
#
# #
# # corpus = TruecaserCorpus(corpus)
# # for sentence in corpus.dev:
# #
# #     # go through each token of sentence
# #     for token in sentence:
# #         # print what you need (text and NER value)
# #         print(f"{token.text}\t{token.get_tag('case').value}")
# #         # print(f"{token.text}")
# #
# #     # print newline at end of each sentence
# #     print()
# #
# # for token in test:
# #     print(token)
# #     print(token.embedding)
from flair.data import Corpus, Sentence, Token
from flair.datasets import CONLL_03
from flair.models import SequenceTagger
from pytest import collect
import mlflow
import mlflow.pytorch
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

corpus: Corpus = WIKINER_ENGLISH(base_path='data/base_word', tag_to_bioes=None, in_memory=True)
categories: Categories = BaseWordCategories()
#
#
# # 3. make the tag dictionary from the corpus
# # tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
tag_dictionary = categories
# # initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # GloVe embeddings
    # WordEmbeddings('glove'),
    ELMoEmbeddings(),
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
    # BertEmbeddings('bert-base-uncased'),
    # CharacterEmbeddings(),

    # # contextual string embeddings, forward
    # PooledFlairEmbeddings('news-forward', pooling='min'),
    #
    # # contextual string embeddings, backward
    # PooledFlairEmbeddings('news-backward', pooling='min'),
]
now = datetime.now()
# print(now)
# embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
embeddings = TransformerWordEmbeddings('bert-base-uncased', fine_tune=True, layers="-1")

#
# # initialize sequence tagger
from flair.models import SequenceTagger

print(embeddings.name)

tagger: SequenceTagger = SequenceTagger(  # hidden_size=256,512
    hidden_size=256,
    rnn_layers=2,
    rnn_type='LSTM',
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type='case',
    use_rnn=False
)

from experiment import Experiment

# with mlflow.start_run() as run:
#     mlflow.log_param("epochs", 5)
#     mlflow.pytorch.log_model(tagger, "models")

experiment: Experiment = Experiment(categories=categories,
                                    embeddings=embedding_types,
                                    corpus=corpus,
                                    tagger=tagger,
                                    max_epochs=5,
                                    base_path='result/wiki_ner/base_word/bert-base-uncased-CRF'
                                    )
experiment.train()
