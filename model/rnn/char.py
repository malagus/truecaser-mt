import logging
from pathlib import Path
from typing import Union, List, Callable, Optional

import flair
import numpy as np
import torch
import torch.nn.functional as F
from flair.data import Dictionary, Sentence, Token, space_tokenizer, Label, DataPoint
from flair.datasets import SentenceDataset, StringDataset
from flair.training_utils import Result, store_embeddings, Metric
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from category import BaseCharCategories
from model.base import BaseTruecaser

log = logging.getLogger("truecaser")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


class CharRNNTrueceser(BaseTruecaser):

    def __init__(
            self,

            embedding_dim=25,
            embedding_dropout=0.0,
            char_dictionary: Dictionary = Dictionary.load("common-chars"),

            tag_dictionary: Dictionary = BaseCharCategories(),

            rnn_layer: int = 1,
            rnn_hidden_size=25,
            rnn_type: str = "LSTM",
            pickle_module: str = "pickle",
            *args, **kwargs
    ):
        super(CharRNNTrueceser, self).__init__()
        # dictionaries
        self.char_dictionary = char_dictionary
        self.tag_dictionary = tag_dictionary
        # char embedding
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout

        self.embedding = nn.Embedding(len(self.char_dictionary.item2idx), self.embedding_dim)
        self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)

        # RNN initialization part
        self.use_rnn = True
        self.rnn_type = rnn_type
        self.rnn_layer = rnn_layer
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_input_dim: int = self.embedding_dim
        self.train_initial_hidden_state = False
        self.bidirectional = True
        # bidirectional LSTM on top of embedding layer
        if self.use_rnn:
            num_directions = 2 if self.bidirectional else 1

            if self.rnn_type in ["LSTM", "GRU"]:
                self.rnn = getattr(torch.nn, self.rnn_type)(
                    self.rnn_input_dim,
                    self.rnn_hidden_size,
                    num_layers=self.rnn_layer,
                    dropout=0.0 if self.rnn_layer == 1 else 0.5,
                    bidirectional=True,
                    batch_first=True,
                )
                # Create initial hidden state and initialize it
                # if self.train_initial_hidden_state:
                #     self.hs_initializer = torch.nn.init.xavier_normal_
                #
                #     self.lstm_init_h = Parameter(
                #         torch.randn(self.nlayers * num_directions, self.hidden_size),
                #         requires_grad=True,
                #     )
                #
                #     self.lstm_init_c = Parameter(
                #         torch.randn(self.nlayers * num_directions, self.hidden_size),
                #         requires_grad=True,
                #     )
                #
                #     # TODO: Decide how to initialize the hidden state variables
                #     # self.hs_initializer(self.lstm_init_h)
                #     # self.hs_initializer(self.lstm_init_c)

            # final linear map to tag space
            self.linear = torch.nn.Linear(
                self.rnn_hidden_size * num_directions, len(self.tag_dictionary)
            )
        else:
            self.linear = torch.nn.Linear(
                self.self.embedding_dim, len(tag_dictionary)
            )
        self.use_crf = False
        self.loss_weights = None

        self.to(flair.device)

    def forward(self, sentences: List[Sentence]):
        torch.set_printoptions(profile="full")
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            longest_token_sequence_in_batch,
            dtype=torch.long,
            device=flair.device,
        )

        sentences_bach = []
        for sentence in sentences:
            char_indices = []

            for token in sentence.tokens:
                char_indices.append(self.char_dictionary.get_idx_for_item(token.text))
            t = torch.tensor(
                char_indices, dtype=torch.long, device=flair.device
            )
            sentences_bach.append(t)
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)
            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    :  nb_padding_tokens
                    ]
                sentences_bach.append(t)
        sentence_tensor = torch.cat(sentences_bach).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
            ]
        )
        #print(sentence_tensor, sentence_tensor.shape)

        sentence_tensor = self.embedding(sentence_tensor)
        #print(sentence_tensor, sentence_tensor.shape)
        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sentence_tensor, lengths, enforce_sorted=False, batch_first=True
            )

            # if initial hidden state is trainable, use this state
            if self.train_initial_hidden_state:
                initial_hidden_state = [
                    self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
                    self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
                ]
                rnn_output, hidden = self.rnn(packed, initial_hidden_state)
            else:
                rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )

        #print(sentence_tensor, sentence_tensor.shape)

        features = self.linear(sentence_tensor)
        #print(features, features.shape)

        return features

    def _calculate_loss(
            self, features: torch.tensor, sentences: List[Sentence]
    ) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tag_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag)

        if self.use_crf:
            pass
            # # pad tags if using batch-CRF decoder
            # tags, _ = pad_tensors(tag_list)
            #
            # forward_score = self._forward_alg(features, lengths)
            # gold_score = self._score_sentence(features, tags, lengths)
            #
            # score = forward_score - gold_score
            #
            # return score.mean()

        else:
            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(
                    features, tag_list, lengths
            ):
                sentence_feats = sentence_feats[:sentence_length]
                score += torch.nn.functional.cross_entropy(
                    sentence_feats, sentence_tags, weight=self.loss_weights
                )
            score /= len(features)
            return score

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence, List[str], str],
            mini_batch_size=32,
            embedding_storage_mode="none",
            all_tag_prob: bool = False,
            verbose: bool = False,
            use_tokenizer: Union[bool, Callable[[str], List[Token]]] = space_tokenizer,
    ) -> List[Sentence]:
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a string or a List of Sentence or a List of string.
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param embedding_storage_mode: 'none' for the minimum memory footprint, 'cpu' to store embeddings in Ram,
        'gpu' to store embeddings in GPU memory.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param use_tokenizer: a custom tokenizer when string are provided (default is space based tokenizer).
        :return: List of Sentence enriched by the predicted tags
        """
        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, Sentence) or isinstance(sentences, str):
                sentences = [sentences]

            if (flair.device.type == "cuda") and embedding_storage_mode == "cpu":
                log.warning(
                    "You are inferring on GPU with parameter 'embedding_storage_mode' set to 'cpu'."
                    "This option will slow down your inference, usually 'none' (default value) "
                    "is a better choice."
                )

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )
            original_order_index = sorted(
                range(len(rev_order_len_index)), key=lambda k: rev_order_len_index[k]
            )

            reordered_sentences: List[Union[Sentence, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            if isinstance(sentences[0], Sentence):
                # remove previous embeddings
                store_embeddings(reordered_sentences, "none")
                dataset = SentenceDataset(reordered_sentences)
            else:
                dataset = StringDataset(
                    reordered_sentences, use_tokenizer=use_tokenizer
                )
            dataloader = DataLoader(
                dataset=dataset, batch_size=mini_batch_size, collate_fn=lambda x: x
            )

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            results: List[Sentence] = []
            for i, batch in enumerate(dataloader):

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {i}")
                results += batch
                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                feature: torch.Tensor = self.forward(batch)
                tags, all_tags = self._obtain_labels(
                    feature=feature,
                    batch_sentences=batch,
                    transitions=transitions,
                    get_all_tags=all_tag_prob,
                )

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token.add_tag_label(self.tag_type, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(self.tag_type, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            results: List[Union[Sentence, str]] = [
                results[index] for index in original_order_index
            ]
            assert len(sentences) == len(results)
            return results

    def evaluate(
            self,
            data_loader: DataLoader,
            out_path: Path = None,
            embedding_storage_mode: str = "none",
    ) -> (Result, float):

        if type(out_path) == str:
            out_path = Path(out_path)

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0

            metric = Metric("Evaluation")

            lines: List[str] = []

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            for batch in data_loader:
                batch_no += 1

                with torch.no_grad():
                    features = self.forward(batch)
                    loss = self._calculate_loss(features, batch)
                    tags, _ = self._obtain_labels(
                        feature=features,
                        batch_sentences=batch,
                        transitions=transitions,
                        get_all_tags=False,
                    )

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label("predicted", tag)

                        # append both to file for evaluation
                        eval_line = "{} {} {} {}\n".format(
                            token.text,
                            token.get_tag(self.tag_type).value,
                            tag.value,
                            tag.score,
                        )
                        lines.append(eval_line)
                    lines.append("\n")
                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
                    ]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.add_tp(tag)
                        else:
                            metric.add_fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.add_fn(tag)
                        else:
                            metric.add_tn(tag)

                store_embeddings(batch, embedding_storage_mode)

            eval_loss /= batch_no

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.micro_avg_f_score(),
                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            return result, eval_loss

    def _obtain_labels(
            self,
            feature: torch.Tensor,
            batch_sentences: List[Sentence],
            transitions: Optional[np.ndarray],
            get_all_tags: bool,
    ) -> (List[List[Label]], List[List[List[Label]]]):
        """
        Returns a tuple of two lists:
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token
           in a sentence for all sentences.
        """

        lengths: List[int] = [len(sentence.tokens) for sentence in batch_sentences]

        tags = []
        all_tags = []
        feature = feature.cpu()
        if self.use_crf:
            feature = feature.numpy()
        else:
            for index, length in enumerate(lengths):
                feature[index, length:] = 0
            softmax_batch = F.softmax(feature, dim=2).cpu()
            scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
            feature = zip(softmax_batch, scores_batch, prediction_batch)

        for feats, length in zip(feature, lengths):
            if self.use_crf:
                confidences, tag_seq, scores = self._viterbi_decode(
                    feats=feats[:length],
                    transitions=transitions,
                    all_scores=get_all_tags,
                )
            else:
                softmax, score, prediction = feats
                confidences = score[:length].tolist()
                tag_seq = prediction[:length].tolist()
                scores = softmax[:length].tolist()

            tags.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(tag), conf)
                    for conf, tag in zip(confidences, tag_seq)
                ]
            )

            if get_all_tags:
                all_tags.append(
                    [
                        [
                            Label(
                                self.tag_dictionary.get_item_for_index(score_id), score
                            )
                            for score_id, score in enumerate(score_dist)
                        ]
                        for score_dist in scores
                    ]
                )

        return tags, all_tags

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embedding_dim": self.embedding_dim,
            "embedding_dropout":self.embedding_dropout,
            "char_dictionary": self.char_dictionary,
            "tag_dictionary": self.tag_dictionary,
            "rnn_layer": self.rnn_layer,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_type": self.rnn_type,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        model = CharRNNTrueceser(
            embedding_dim=state["embedding_dim"],
            embedding_dropout=state["embedding_dropout"],
            char_dictionary=state["char_dictionary"],

            tag_dictionary=state["tag_dictionary"],

            rnn_layer=state["rnn_layer"],
            rnn_hidden_size=state["rnn_hidden_size"],
            rnn_type=state["rnn_type"],

        )
        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    def _fetch_model(model_name) -> str:
        return model_name
