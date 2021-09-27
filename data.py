from typing import List

from flair.data import Token, Sentence, Corpus
from flair.datasets import SentenceDataset

from category import Categories, BaseCharCategories


def character_tokenizer(text: str) -> List[Token]:
    """
    Tokenizer based on characters.
    """
    tokens: List[Token] = []
    for index, char in enumerate(text):
        tokens.append(
            Token(
                text=char, start_position=index
            )
        )

    return tokens


class TruecaserSentence(Sentence):
    def __init__(
            self,
            original_sentence: Sentence,
            categories: Categories = BaseCharCategories(),
            lower_case: bool = True
    ):
        super(TruecaserSentence, self).__init__(
            text=original_sentence.to_plain_string(),
            labels=original_sentence.labels,
            use_tokenizer=character_tokenizer,
            language_code=original_sentence.language_code
        )
        self.original_sentence = original_sentence
        for token in self.tokens:
            token.add_tag(
                'case', categories.encode(token.text)
            )
            if lower_case:
                token.text = token.text.lower()


class TruecaserCorpus(Corpus):
    def __init__(self, corpus: Corpus):
        dev = SentenceDataset(list((TruecaserSentence(sentence) for sentence in corpus.dev)))
        train = SentenceDataset(list((TruecaserSentence(sentence) for sentence in corpus.train)))
        test = SentenceDataset(list((TruecaserSentence(sentence) for sentence in corpus.test)))

        super(TruecaserCorpus, self).__init__(train, dev, test, corpus.name + '_truecase')
