import pandas as pd 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments



import torch.nn as nn
import torch.nn.functional as F

import argparse
import logging
import os
import random
import sys

from datetime import datetime

if 'SM_MODEL_DIR' in os.environ:
    pass
else:
    os.environ["SM_MODEL_DIR"] = "./"
    os.environ["SM_CHANNEL_TRAIN"] = "./"



def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--model_id", type=str, default="google/flan-t5-small")
    parser.add_argument("--learning_rate", type=float, default=0.00005)

    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    args, _ = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    PRETRAINED_MODEL = args.model_id 
    BATCH_SIZE = args.train_batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    DATASET_LOCATION = args.training_dir
    MODEL_SAVE_LOCATION = args.output_dir
    
    dataset = load_from_disk(DATASET_LOCATION)
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    print("Start training ...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer of FLAN-t5-base
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    # Generate the transfomer input token for training 
    max_source_length = 512
    max_target_length = 128    

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
    print(tokenized_dataset)

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer,model=model,label_pad_token_id=label_pad_token_id,pad_to_multiple_of=8)

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_SAVE_LOCATION,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        # logging & evaluation strategies
        logging_dir=f"{MODEL_SAVE_LOCATION}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1"
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        #compute_metrics=compute_metrics,
    )
    
    # Start training
    trainer.train()
