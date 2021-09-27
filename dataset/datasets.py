import logging
import re
from pathlib import Path
from typing import Union, Dict, List

import flair
from flair.data import Corpus, FlairDataset, Sentence, Token
from flair.datasets import ColumnCorpus
from nltk import sent_tokenize, word_tokenize
from torch.utils.data.dataset import random_split

from category import Categories, BaseWordCategories, BaseCharCategories
from dataset.processor import character_tokenizer

log = logging.getLogger("truecaser")


class TruecaseDataset(FlairDataset):
    def __init__(
            self,
            dataset: FlairDataset,
            lower_text=True,
            categories: Categories = BaseCharCategories()
    ):
        self.dataset = dataset
        self.lower_text = lower_text
        self.use_char_tokenizer = isinstance(categories, BaseCharCategories)
        self.categories = categories

    def is_in_memory(self) -> bool:
        return False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int = 0) -> Sentence:
        sentence: Sentence = self.dataset[index]
        new_sentence: Sentence
        if self.use_char_tokenizer:
            new_sentence = Sentence(text=sentence.to_plain_string(), use_tokenizer=character_tokenizer)
        else:
            new_sentence = Sentence()
            new_sentence.tokens = sentence.tokens.copy()
        for token in new_sentence.tokens:
            token.add_tag(
                'case', self.categories.encode(token.text)
            )
            if self.lower_text:
                token.text = token.text.lower()
        return new_sentence


class LineCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            column_format: Dict[int, str],
            train_file=None,
            test_file=None,
            dev_file=None,
            tag_to_bioes=None,
            comment_symbol: str = None,
            in_memory: bool = True,
            encoding: str = "utf-8",
            document_separator_token: str = None,
    ):
        """
        Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param document_separator_token: If provided, multiple sentences are read into one object. Provide the string token
        that indicates that a new document begins
        :return: a Corpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith(".gz"):
                    continue
                if "train" in file_name and not "54019" in file_name:
                    train_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

            # if no test file is found, take any file with 'test' in name
            if test_file is None:
                for file in data_folder.iterdir():
                    file_name = file.name
                    if file_name.endswith(".gz"):
                        continue
                    if "test" in file_name:
                        test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        # get train data
        train = LineDataset(
            train_file,
            column_format,
            tag_to_bioes,
            encoding=encoding,
            comment_symbol=comment_symbol,
            in_memory=in_memory,
            document_separator_token=document_separator_token,
        )

        # read in test file if exists, otherwise sample 10% of train data as test dataset
        if test_file is not None:
            test = LineDataset(
                test_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
            )
        else:
            train_length = len(train)
            test_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - test_size, test_size])
            train = splits[0]
            test = splits[1]

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
        if dev_file is not None:
            dev = LineDataset(
                dev_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
            )
        else:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]

        super(LineCorpus, self).__init__(train, dev, test, name=data_folder.name)


class LineDataset(FlairDataset):
    def __init__(
            self,
            path_to_line_file: Path,
            column_name_map: Dict[int, str],
            tag_to_bioes: str = None,
            comment_symbol: str = None,
            in_memory: bool = True,
            document_separator_token: str = None,
            encoding: str = "utf-8",
            categories: Categories = BaseWordCategories()
    ):
        """
        Instantiates a line dataset

        :param path_to_line_file: path to the file with the line-formatted data
        :param column_name_map: a map specifying the column format
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param document_separator_token: If provided, multiple sentences are read into one object. Provide the string token
        that indicates that a new document begins
        :param categories token categories
        """
        self.categories = categories
        assert path_to_line_file.exists()
        self.path_to_column_file = path_to_line_file
        self.tag_to_bioes = tag_to_bioes
        self.column_name_map = column_name_map
        self.comment_symbol = comment_symbol
        self.document_separator_token = document_separator_token

        # store either Sentence objects in memory, or only file offsets
        self.in_memory = in_memory
        if self.in_memory:
            self.sentences: List[Sentence] = []
        else:
            self.indices: List[int] = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_column: int = 0
        for column in self.column_name_map:
            if column_name_map[column] == "text":
                self.text_column = column

        # determine encoding of text file
        self.encoding = encoding

        sentence: Sentence = Sentence()
        with open(str(self.path_to_column_file), encoding=self.encoding) as f:

            line = f.readline()
            position = 0

            while line:

                if self.comment_symbol is not None and line.startswith(comment_symbol):
                    line = f.readline()
                    continue

                # if self.__line_completes_sentence(line):
                #
                #     if len(sentence) > 0:
                #
                #         sentence.infer_space_after()
                #         if self.in_memory:
                #             if self.tag_to_bioes is not None:
                #                 sentence.convert_tag_scheme(
                #                     tag_type=self.tag_to_bioes, target_scheme="iobes"
                #                 )
                #             self.sentences.append(sentence)
                #         else:
                #             self.indices.append(position)
                #             position = f.tell()
                #         self.total_sentence_count += 1
                #     sentence: Sentence = Sentence()
                #
                # else:
                #     fields: List[str] = re.split("\s+", line)
                #     token = Token(fields[self.text_column])
                #     for column in column_name_map:
                #         if len(fields) > column:
                #             if column != self.text_column:
                #                 token.add_tag(
                #                     self.column_name_map[column], fields[column]
                #                 )
                #
                #     if not line.isspace():
                #         sentence.add_token(token)
                for sent in sent_tokenize(line):
                    sentence = Sentence()
                    for tok in word_tokenize(sent):
                        token = Token(tok.lower())
                        token.add_tag(
                            'case', self.categories.encode(tok)
                        )
                        sentence.add_token(token)
                    self.sentences.append(sentence)

                line = f.readline()

        # if len(sentence.tokens) > 0:
        #     sentence.infer_space_after()
        #     if self.in_memory:
        #         self.sentences.append(sentence)
        #     else:
        #         self.indices.append(position)
        #     self.total_sentence_count += 1

    def __line_completes_sentence(self, line: str) -> bool:
        sentence_completed = line.isspace()
        if self.document_separator_token:
            sentence_completed = False
            fields: List[str] = re.split("\s+", line)
            if len(fields) >= self.text_column:
                if fields[self.text_column] == self.document_separator_token:
                    sentence_completed = True
        return sentence_completed

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        if self.in_memory:
            sentence = self.sentences[index]

        # else:
        #     with open(str(self.path_to_column_file), encoding=self.encoding) as file:
        #         file.seek(self.indices[index])
        #         line = file.readline()
        #         sentence: Sentence = Sentence()
        #         while line:
        #             if self.comment_symbol is not None and line.startswith(
        #                     self.comment_symbol
        #             ):
        #                 line = file.readline()
        #                 continue
        #
        #             if self.__line_completes_sentence(line):
        #                 if len(sentence) > 0:
        #                     sentence.infer_space_after()
        #                     if self.tag_to_bioes is not None:
        #                         sentence.convert_tag_scheme(
        #                             tag_type=self.tag_to_bioes, target_scheme="iobes"
        #                         )
        #                     return sentence
        #
        #             else:
        #                 fields: List[str] = re.split("\s+", line)
        #                 token = Token(fields[self.text_column])
        #                 for column in self.column_name_map:
        #                     if len(fields) > column:
        #                         if column != self.text_column:
        #                             token.add_tag(
        #                                 self.column_name_map[column], fields[column]
        #                             )
        #
        #                 if not line.isspace():
        #                     sentence.add_token(token)
        #
        #             line = file.readline()
        return sentence


class Conll_2003_Trucase(LineCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            document_as_sequence: bool = False,
    ):
        """
        :param base_path: Path to corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "np", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: dataset not found at "{data_folder}".')
            log.warning("-" * 100)

        super(Conll_2003_Trucase, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            test_file='test_docs.txt',
            train_file='train_docs.txt',
            dev_file='valid_docs.txt',
        )


class Wiki_Data(LineCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            document_as_sequence: bool = False,
    ):
        """
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "np", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG:dataset not found at "{data_folder}".')
            log.warning("-" * 100)

        super(Wiki_Data, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            test_file='test.txt',
            train_file='train.txt',
            dev_file='dev.txt',
        )


class WIKINER_ENGLISH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner", 3: "case"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        #_download_wikiner("en", dataset_name)

        super(WIKINER_ENGLISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )
