import pandas as pd
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np

import re
import sys
import os

# Model to use
MODEL_NAME = 'None'

# directories of the datasets to use
train_dataset = 'None'
test_dataset = 'None'

# directory to save the trained model
save_directory = 'None'
save_flag = False

argv = sys.argv
argc = len(argv)
# TODO: Use regex instead
if(argc > 9 or argc < 7):
  print("Usage: python SentimentTweets.py --train_data <directory> --test_data <directory> --model_name <name_or_path> [--save_model_on_directory <directory>]")
  sys.exit(-1)
 
else:
  MODEL_NAME = argv[6]
  train_dataset = argv[2]
  test_dataset = argv[4]
  if(argc == 9):
    save_directory = argv[8]
    save_flag = True


# Load the MODEL_NAME tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

# Load the MODEL_NAME model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# max tensor length
max_length = max_length=model.config.max_position_embeddings
max_length = max_length if max_length < 400 else 400


def create_datasets(df_train, df_test):
  ds_train = Dataset.from_pandas(df_train)
  ds_test = Dataset.from_pandas(df_test)
  return ds_train, ds_test


# Create a function to tokenize a set of texts
def preprocessing_tokenizer(input):
  return tokenizer(input["text"], truncation=True, padding=True, max_length=max_length)


def preprocess(ds_train, ds_test):
  preprocessed_train = ds_train.map(preprocessing_tokenizer, batched=True,load_from_cache_file=False)
  preprocessed_test = ds_test.map(preprocessing_tokenizer, batched=True,load_from_cache_file=False)
  preprocessed_train.set_format(type='torch', columns=['input_ids', 'label', 'attention_mask'])
  preprocessed_test.set_format(type='torch', columns=['input_ids', 'label', 'attention_mask'])
  return preprocessed_train, preprocessed_test


def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def myTrainer(preprocessed_train, preprocessed_test, save_model=False):
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  training_args = TrainingArguments(
    output_dir=save_directory,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
  ) if save_model else TrainingArguments(
    output_dir=".",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed_train,
    eval_dataset=preprocessed_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
  )
  return trainer


def fineTunning(trainData, testData, save_model = False):
  ds_train, ds_test = create_datasets(trainData, testData)
  preprocessed_train, preprocessed_test = preprocess(ds_train, ds_test)

  trainer = myTrainer(preprocessed_train, preprocessed_test, save_model=save_model)
  metrics = trainer.train()
  print(metrics)
  
  if(save_model):
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    pass

  return trainer


df_train = pd.read_csv(train_dataset, sep='\t')
df_test = pd.read_csv(test_dataset, sep='\t')

trainer = fineTunning(df_train, df_test, save_model=save_flag)