from abc import abstractmethod, ABCMeta
from typing import Dict

from flair.data import Dictionary


class Categories(Dictionary):
    """
        Abstract class for all definitions of trucateser model categories.

    """

    def __init__(self):
        super(Categories, self).__init__(add_unk=False)

    @abstractmethod
    def encode(self, x: str) -> str:
        """Map word/char into specific category."""

    @abstractmethod
    def decode(self, x: str, category: str) -> str:
        """Based on word/char and category return trucase value of it."""


class BaseWordCategories(Categories):
    """
        Base definition of categories for truecaser mechanism with fallowing categories:
            * "L" (0) - word start with lowercase character
            * "U" (1) - word start with uppercase character
            * "U_ALL" (2) - all character are capitalised or capitalization is noisy
    """

    def __init__(self):
        super(BaseWordCategories, self).__init__()
        self.add_item("L")
        self.add_item("U")
        self.add_item("U_ALL")

    def encode(self, word: str) -> str:
        if word.isupper():
            return "U_ALL"
        elif word.islower():
            return "L"
        elif word[0].isupper() and word[1:].islower():
            return "U"
        else:
            return "U_ALL"

    def decode(self, word: str, category: str) -> str:
        if category == "L":
            return word.lower()
        elif category == "U":
            return word[0].upper() + word[1:]
        else:
            return word.upper()
            # # hack for non english letters
            # return word.upper().encode('ascii', 'ignore').decode("utf-8")


class NoisyWordCategories(Categories):
    """
        Base definition of categories for truecaser mechanism with fallowing categories:
            * "L" (0) - word start with lowercase character
            * "U" (1) - word start with uppercase character
            * "C" (2) - word with all uppercase characters
            * "M" (3) - capitalisation is mixed (BlackBery etc.)
    """

    def __init__(self):
        super(NoisyWordCategories, self).__init__()
        self.add_item("L")
        self.add_item("U")
        self.add_item("U_ALL")
        self.add_item("M")

    def encode(self, word: str) -> str:
        if word.isupper():
            return "U_ALL"
        elif word.islower():
            return "L"
        elif word[0].isupper() and word[1:].islower():
            return "U"
        else:
            return "M"

    def decode(self, word: str, category: str) -> str:
        if category == "L":
            return word.lower()
        elif category == "U":
            return word[0].upper() + word[1:]
        elif category == "U_ALL":
            return word.upper()
        else:
            # implement mixed
            return word


class BaseCharCategories(Categories):
    """
        Base definition of categories for truecaser mechanism with fallowing categories:
            * "L" (0) - word start with lowercase character
            * "U" (1) - word start with uppercase character
    """

    def __init__(self):
        super(BaseCharCategories, self).__init__()
        self.add_item("L")
        self.add_item("U")

    def encode(self, char: str) -> str:
        if char.isupper():
            return "U"
        elif char.islower():
            return "L"
        else:
            return "O"

    def decode(self, char: str, category: str) -> str:
        if category == "L":
            return char.lower()
        else:
            return char.upper()
