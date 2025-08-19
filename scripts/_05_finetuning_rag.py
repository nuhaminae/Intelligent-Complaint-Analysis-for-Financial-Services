# _05_finetuning_rag.py

import json
import os
import warnings

import numpy as np
from datasets import Dataset
from evaluate import load
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

warnings.simplefilter("ignore", category=FutureWarning)


class FineTune:
    def __init__(self, model_name, train_pair_path, output_dir):
        """Fine-tuning pipeline for a sequence-to-sequence model.

        Args:
            model_name (str): The name of the pre-trained model to fine-tune.
            train_pair_path (str): The path to the training data (JSONL format).
            output_dir (str): The directory to save the fine-tuned model.
        """
        self.model_name = model_name
        self.train_pair_path = train_pair_path
        self.output_dir = output_dir

        # Create directories if they do not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("🚀 Initialising fine-tuning pipeline...")
        self.tokeniser = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def safe_relpath(self, path, start=None):
        """
        Return a relative path, handling cases where paths are on different drives.

        Args:
            path (str): The path to make relative.
            start (str, optional): The starting directory.
                                    Defaults to current working directory.

        Returns:
            str: The relative path if possible, otherwise the original path.
        """
        try:
            return os.path.relpath(path, start)
        except ValueError:
            # Fallback to absolute path if on different drives
            return path

    def load_jsonl(self):
        """Load training pairs from a JSONL file.

        Returns:
            list: A list of training pairs.
        """
        with open(self.train_pair_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        print(
            f"\n📥 Loaded {len(data)} training pairs from {self.safe_relpath(self.train_pair_path)}"
        )
        return data

    def tokenise_function(self, examples):
        """Tokenise input and output texts for the model.

        Args:
            examples (dict): A dictionary containing "input" and "output" keys.

        Returns:
            dict: A dictionary containing tokenised "input_ids" and "attention_mask".
        """
        # Tokenise inputs
        prompts = [
            f"{instr} {inp}"
            for instr, inp in zip(examples["instruction"], examples["input"])
        ]
        return self.tokeniser(
            prompts,
            text_target=examples["output"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def compute_metrics(self, eval_pred: EvalPrediction):
        """Compute evaluation metrics for the model.

        Args:
            eval_pred (EvalPrediction): The evaluation predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        rouge = load("rouge")

        # Convert logits to token IDs
        if isinstance(predictions, tuple):  # Some models return (logits, ...)
            predictions = predictions[0]
        pred_ids = np.argmax(predictions, axis=-1)

        decoded_preds = self.tokeniser.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = self.tokeniser.batch_decode(labels, skip_special_tokens=True)

        return rouge.compute(predictions=decoded_preds, references=decoded_labels)

    def train(self):
        """Fine-tune the model on the loaded dataset."""
        raw_data = self.load_jsonl()
        dataset = Dataset.from_list(raw_data)

        # Split 80% train, 20% test
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"\n📊 Train dataset size: {len(train_dataset)}")
        print(f"📊 Eval dataset size: {len(eval_dataset)}")

        print("\n📦 Tokenising datasets...")
        # tokenised_train = train_dataset.map(self.tokenise_function)
        tokenised_train = train_dataset.map(
            self.tokenise_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        tokenised_eval = eval_dataset.map(
            self.tokenise_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
        # tokenised_eval = eval_dataset.map(self.tokenise_function)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=8,
            num_train_epochs=4,
            logging_strategy="steps",
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenised_train,
            eval_dataset=tokenised_eval,
            tokenizer=self.tokeniser,
            compute_metrics=self.compute_metrics,
        )

        print("\n🧠 Starting training...")
        trainer.train()
        trainer.save_model()
        metrics = trainer.evaluate()
        print("\n📈 Final Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print(
            f"\n💾 Model fine-tuned and saved to {self.safe_relpath(self.output_dir)}"
        )
