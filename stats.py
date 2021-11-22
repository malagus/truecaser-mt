from flair.data import Corpus
from flair.datasets import WIKINER_ENGLISH, CONLL_03

from dataset.datasets import Conll_2003_Trucase, TruecaseDataset
# Datasets statistics
corpus: Corpus = CONLL_03(base_path='data')
print(corpus)
stats = corpus.obtain_statistics()
print(stats)
corpus._train = TruecaseDataset(corpus.train)
corpus._dev = TruecaseDataset(corpus.dev)
corpus._test = TruecaseDataset(corpus.test)
stats = corpus.obtain_statistics()
print(stats)
corpus: Corpus = WIKINER_ENGLISH(base_path='data', tag_to_bioes=None)
print(corpus)
stats = corpus.obtain_statistics()
print(stats)
corpus._train = TruecaseDataset(corpus.train)
corpus._dev = TruecaseDataset(corpus.dev)
corpus._test = TruecaseDataset(corpus.test)
stats = corpus.obtain_statistics()
print(stats)

from datasets import load_dataset
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
for task in task_to_keys.keys():
    dataset = load_dataset('glue', task)