from typing import List

import nltk
from flair.data import Token, Sentence, Corpus, FlairDataset
from flair.datasets import SentenceDataset

from category import Categories


def character_tokenizer(text: str) -> List[Token]:
    """
    Tokenizer based on characters.
    """
    tokens: List[Token] = []
    for index, char in enumerate(text):
        tokens.append(
            Token(
                text=char
            )
        )

    return tokens


class DatasetProcessor(object):
    def __init__(self, lower_text=True, use_char_tokenizer=False, categories: Categories = None):

        self.lower_text = lower_text
        self.use_char_tokenizer = use_char_tokenizer
        self.categories = categories

    def check_sentence_sanity(self, sentence: Sentence):
        """ Checks the sanity of the sentence. If the sentence is for example all uppercase, it is recjected"""
        if sentence.to_plain_string() == '-DOCSTART-':
            return False
        # caseDist = nltk.FreqDist()
        #
        # for token in sentence.tokens:
        #     caseDist[self.categories.encode(token.text)] += 1
        #
        # if caseDist.most_common(1)[0][0] == 'U_ALL':
        #     print(caseDist)
        #     return False

        return True

    def __process_sentence(self, sentence: Sentence):
        if not self.check_sentence_sanity(sentence):
            return None
        new_sentence: Sentence
        if self.use_char_tokenizer:
            new_sentence = Sentence(text=sentence.to_plain_string(), use_tokenizer=character_tokenizer)
        else:
            new_sentence = Sentence()
            new_sentence.tokens = sentence.tokens.copy()

        for token in new_sentence.tokens:
            if self.categories is not None:
                token.add_tag(
                    'case', self.categories.encode(token.text)
                )
            if self.lower_text:
                token.text = token.text.lower()
        return new_sentence

    def process(self, dataset: FlairDataset):
        sentences = list(filter(None, map(self.__process_sentence, dataset)))
        return SentenceDataset(sentences)
