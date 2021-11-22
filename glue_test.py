import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import wandb
from datasets import load_dataset, load_metric, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from wandb.wandb_run import Run

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
# Glue best models
tasks = {
    "cola": {
        'cased': 'wandb/run-20201121_220000-8phbczo8',
        'uncased': 'wandb/run-20201122_132921-eguxjqiz'
    },
    "mnli": {
        'cased': 'wandb/run-20201121_222446-i5xikd7a',
        'uncased': 'wandb/run-20201122_133847-j4mgp0ae'
    },
    "mrpc": {
        'cased': 'wandb/run-20201122_085041-tb3ew7s7',
        'uncased': 'wandb/run-20201122_162934-xj0944ug'
    },
    "qnli": {
        'cased': 'wandb/run-20201122_085719-qv1leti9',
        'uncased': 'wandb/run-20201122_163600-pultbbkc'
    },
    "qqp": {
        'cased': 'wandb/run-20201122_094826-y06ft1a6',
        'uncased': 'wandb/run-20201122_172614-qinpv5xg'
    },
    "rte": {
        'cased': 'wandb/run-20201122_124032-vlcl0q68',
        'uncased': 'wandb/run-20201122_204557-x5mmzibr'
    },
    "sst2": {
        'cased': 'wandb/run-20201122_124625-04rvzlyk',
        'uncased': 'wandb/run-20201122_205230-wyiykfyv'
    },
    "stsb": {
        'cased': 'wandb/run-20201122_131710-rzidw0c3',
        'uncased': 'wandb/run-20201122_213731-zfy6ehgb'
    },
    "wnli": {
        'cased': 'wandb/run-20201122_132421-1yxwh8hd',
        'uncased': 'wandb/run-20201122_214734-4ah76374'
    },

}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

# Modified script for glue testing
def main(task_name, model_name, case):
    output_dir = 'test/glue'
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # # Set seed before initializing model.
    # set_seed(training_args.seed)
    #
    #
    if task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", task_name)

    # Labels
    if task_name is not None:
        is_regression = task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1

    # # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task=task_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    #
    # # Preprocessing the datasets
    if task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length"
    max_length = 128

    # # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and task_name is not None
            and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    datasets_lowercased = load_from_disk('data/glue/' + task_name + '/lowercase').map(preprocess_function, batched=True,
                                                                                      load_from_cache_file=True)
    datasets_truecased = load_from_disk('data/glue/' + task_name + '/truecase').map(preprocess_function, batched=True,
                                                                                    load_from_cache_file=True)

    eval_dataset = datasets["validation_matched" if task_name == "mnli" else "validation"]
    eval_dataset_lowercased = datasets_lowercased["validation_matched" if task_name == "mnli" else "validation"]
    eval_dataset_truecased = datasets_truecased["validation_matched" if task_name == "mnli" else "validation"]

    # Get the metric function
    if task_name is not None:
        metric = load_metric("glue", task_name)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        # tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator
    )

    # Evaluation

    logger.info("*** Evaluate base ***")
    eval_results = {}

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [task_name]
    eval_datasets = [eval_dataset]
    if task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(datasets["validation_mismatched"])

    for eval_dataset, task in zip(eval_datasets, tasks):
        # eval_dataset.remove_columns_("label")
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(output_dir, f"test_results_{task}_{case}.txt")
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** Test results {task} *****")
            for key, value in eval_result.items():
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        eval_results.update(eval_result)
    if case == 'uncased':
        return

    logger.info("*** Evaluate lowercase ***")
    eval_results = {}

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [task_name]
    eval_datasets = [eval_dataset_lowercased]
    if task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(datasets_lowercased["validation_mismatched"])

    for eval_dataset, task in zip(eval_datasets, tasks):
        # eval_dataset.remove_columns_("label")

        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(output_dir, f"test_results_{task}_{case}_lowercase.txt")
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** Test results {task} *****")
            for key, value in eval_result.items():
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        eval_results.update(eval_result)
    logger.info("*** Evaluate base ***")
    eval_results = {}

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [task_name]
    eval_datasets = [eval_dataset_truecased]
    if task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(datasets_truecased["validation_mismatched"])

    for eval_dataset, task in zip(eval_datasets, tasks):
        # eval_dataset.remove_columns_("label")
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(output_dir, f"test_results_{task}_{case}_truecase.txt")
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** Test results {task} *****")
            for key, value in eval_result.items():
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        eval_results.update(eval_result)


# if __name__ == "__main__":
# for tasks_name in tasks.keys():
#     for model in tasks[tasks_name].keys():
#         main(tasks_name, 'wandb/run-20201121_220000-8phbczo8', 'cased')
if __name__ == "__main__":
    for tasks_name in tasks.keys():
        for model in tasks[tasks_name].keys():
            try:
                main(tasks_name, tasks[tasks_name][model], model)
            except Exception as e:
                logger.error(e.message)
