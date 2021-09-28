from datasets import load_dataset
from flair.data import Sentence

from category import BaseWordCategories
from model.sequence_tagger_model import SequenceTagger as TruecaserTagger

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"), #err :/
    # "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def lowercase(task, example):
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence1_key is not None:
        example[sentence1_key] = example[sentence1_key].lower()
    if sentence2_key is not None:
        example[sentence2_key] = example[sentence2_key].lower()
    return example


trucaser: TruecaserTagger = TruecaserTagger.load('best_models/truecaser.pt')
trucaser.eval()
category = BaseWordCategories()


def truecase_text(text):
    sentence = Sentence(text.lower())

    try:
        trucaser.predict(sentence)
        for token in sentence.tokens:
            token.text = category.decode(token.text, token.get_tag('case').value)
        sentence.tokenized = None
    except Exception as e:
        print(e)

    return sentence.to_original_text()


def truecase(task, example):
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence1_key is not None:
        example[sentence1_key] = truecase_text(example[sentence1_key])
    if sentence2_key is not None:
        example[sentence2_key] = truecase_text(example[sentence2_key])

    return example


for task in task_to_keys.keys():
    dataset = load_dataset('glue', task)
    # lowercase_dataset = dataset.map(lambda example: lowercase(task, example))
    # lowercase_dataset.save_to_disk('data/glue/' + task + '/lowercase')
    truecase_dataset = dataset.map(lambda example: truecase(task, example))
    truecase_dataset.save_to_disk('data/glue/' + task + '/truecase')
