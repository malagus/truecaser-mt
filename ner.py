import logging
from pathlib import Path

from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.models import SequenceTagger
from flair.training_utils import add_file_handler

from category import BaseWordCategories
from dataset.datasets import WIKINER_ENGLISH
from dataset.processor import DatasetProcessor
from model.sequence_tagger_model import SequenceTagger as TruecaserTagger

tasks = {
    'ner': {
        'cased': 'wandb/run-20201127_105306-794ghmkd/final-model.pt',
        'uncased': 'wandb/run-20201127_183247-ezvhdgrd/final-model.pt'

    },
    'pos': {
        'cased': 'wandb/run-20201127_143605-8b6vt585/final-model.pt',
        'uncased': 'wandb/run-20201127_224614-mzmq1ggc/final-model.pt'
    }

}

datasets = [
    WIKINER_ENGLISH(base_path='data', tag_to_bioes=None, in_memory=True),
    CONLL_03(base_path='data')
]


def truecase(sentences):
    trucaser: TruecaserTagger = TruecaserTagger.load('best_models/truecaser.pt')
    trucaser.eval()
    category = BaseWordCategories()
    trucaser.predict(sentences)
    for sentence in sentences:
        try:
            # trucaser.predict(sentence)
            # log.info(sentence.to_plain_string())
            for token in sentence.tokens:
                token.text = category.decode(token.text, token.get_tag('case').value)
            sentence.tokenized = None
        except Exception as e:
            log.error(e.message)
        # log.info(sentence.to_plain_string())


log = logging.getLogger("flair")

processor = DatasetProcessor()


def run_test(base_path, corpus, model_path, variant):
    log_handler = add_file_handler(log, base_path / "training.log")
    tagger = SequenceTagger.load(model_path)
    tagger.eval()
    log.info(variant)
    test_results, test_loss = tagger.evaluate(
        corpus.test,
        mini_batch_size=32,
        num_workers=6,
        out_path=base_path / "test.tsv",
        embedding_storage_mode="none",
    )

    log.info(test_results.log_line)
    log.info(test_results.detailed_results)
    log.removeHandler(log_handler)


base_path = Path("test/")


def experiment(base_path, corpus, variant):
    for task_name in tasks.keys():
        task = tasks[task_name]
        for type in task.keys():
            model_path = task[type]
            run_test(base_path / task_name / type / variant, corpus, model_path, variant)


log_handler_global = add_file_handler(log, base_path / "training.log")
for corpus in datasets:
    dataset_name = corpus.__class__.__name__.lower()
    experiment(base_path / dataset_name, corpus, 'original')
    corpus._test = processor.process(corpus.test)
    experiment(base_path / dataset_name, corpus, 'lowercase')
    truecase(corpus.test)
    experiment(base_path / dataset_name, corpus, 'truecase')
