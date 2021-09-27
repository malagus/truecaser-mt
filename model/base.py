import logging
from abc import ABCMeta, abstractmethod
from typing import Union, List

import flair
import torch
from flair.data import Sentence

log = logging.getLogger("truecaser")


class BaseTruecaser(flair.nn.Model, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(BaseTruecaser, self).__init__(*args, **kwargs)
        self.tag_type = 'case'

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        features = self.forward(data_points)
        return self._calculate_loss(features, data_points)

    @abstractmethod
    def _calculate_loss(
            self, features: torch.tensor, sentences: List[Sentence]
    ) -> float:
        pass

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    @staticmethod
    def _filter_empty_string(texts: List[str]) -> List[str]:
        filtered_texts = [text for text in texts if text]
        if len(texts) != len(filtered_texts):
            log.warning(
                f"Ignore {len(texts) - len(filtered_texts)} string(s) with no tokens."
            )
        return filtered_texts


class WordTruecaser(BaseTruecaser, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(WordTruecaser, self).__init__(*args, **kwargs)


class CharTruecaser(BaseTruecaser, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(CharTruecaser, self).__init__(*args, **kwargs)
