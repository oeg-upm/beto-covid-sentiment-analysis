import pandas as pd
import sys

# Split dataset into train (90%) and test (10%)
from sklearn.model_selection import train_test_split

# For setting up the GPU you have to go: Runtime -> Change runtime type -> Hardware accelerator: GPU
import torch

# Installing Hugging Face
import os
import re
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# BERT tokenizer
from transformers import BertTokenizer

# PyTorch DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# BERT classifier
# import torch
import torch.nn as nn
# from transformers import BertModel

# Optimizer & Learning Rate Scheduler
from transformers import AdamW, get_linear_schedule_with_warmup

# Training loop
import random
import time

# Custom Model
from CustomModel import BertClassifier

# Model to use
BETO = 'None'

# directories of the datasets to use
train_dataset = 'None'
test_dataset = 'None'

# directory to save the trained model
save_directory = 'None'
save_flag = False

argv = sys.argv
argc = len(argv)
if(argc > 9 or argc < 7):
  print("Usage: python SentimentTweets.py --train_data <directory> --test_data <directory> --model_name <name_or_path> [--save_model_on_directory <directory>]")
  sys.exit(-1)
 
else:
  BETO = argv[6]
  train_dataset = argv[2]
  test_dataset = argv[4]
  if(argc == 9):
    save_directory = argv[8]
    save_flag = True

def createTrainVal(dataset):
  X = dataset.full_text.values
  y = dataset.sentiment.values

  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2020)

  return X_train, X_val, y_train, y_val

def createTest(dataset):
  X_test = dataset.full_text.values

  return X_test

def useGPU():
  device = ''
  if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

  else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

  return device

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """

    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data, tokenizer, MAX_LEN):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def returnMaxLength(X_train, X_val, X_test, tokenizer):
  # Concatenate train and val
  X = np.concatenate([X_train, X_val])

  # Concatenate all train data and test data
  all_tweets = np.concatenate([X, X_test])

  # Encode our concatenated data
  encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_tweets]

  # Find the maximum length
  max_len = max([len(sent) for sent in encoded_tweets])
  print('Max length: ', max_len)
  return max_len

def createDataloader(X_train, X_val, y_train, y_val, tokenizer, MAX_LEN):
  # Run function `preprocessing_for_bert` on the train set and the validation set
  print('Tokenizing data...')
  train_inputs, train_masks = preprocessing_for_bert(X_train, tokenizer, MAX_LEN)
  val_inputs, val_masks = preprocessing_for_bert(X_val, tokenizer, MAX_LEN)

  # Create PyTorch DataLoader

  # Convert other data types to torch.Tensor
  train_labels = torch.tensor(y_train)
  val_labels = torch.tensor(y_val)

  # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
  batch_size = 32

  # Create the DataLoader for our training set
  train_data = TensorDataset(train_inputs, train_masks, train_labels)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

  # Create the DataLoader for our validation set
  val_data = TensorDataset(val_inputs, val_masks, val_labels)
  val_sampler = SequentialSampler(val_data)
  val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

  return train_dataloader, val_dataloader

def createFullDataloader(X_train, X_val, y_train, y_val, tokenizer, MAX_LEN):
  # Run function `preprocessing_for_bert` on the train set and the validation set
  print('Tokenizing data...')
  train_inputs, train_masks = preprocessing_for_bert(X_train, tokenizer, MAX_LEN)
  val_inputs, val_masks = preprocessing_for_bert(X_val, tokenizer, MAX_LEN)

  # Create PyTorch DataLoader

  # Convert other data types to torch.Tensor
  train_labels = torch.tensor(y_train)
  val_labels = torch.tensor(y_val)

  # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
  batch_size = 32

  # Create the DataLoader for our training set
  train_data = TensorDataset(train_inputs, train_masks, train_labels)

  # Create the DataLoader for our validation set
  val_data = TensorDataset(val_inputs, val_masks, val_labels)

  # Create Dataloader out of the Entire Training Data
  # Concatenate the train set and the validation set
  full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
  full_train_sampler = RandomSampler(full_train_data)
  full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=32)

  return full_train_dataloader


# Optimizer & Learning Rate Scheduler

def initialize_model(device, train_dataloader, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(BETO, freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(device, optimizer, scheduler, model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader, device)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader, device):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def fineTunning(trainData, testData, save_model = False):
  X_train, X_val, y_train, y_val = createTrainVal(trainData)
  X_test = createTest(testData)

  device = useGPU()

  # Load the BERT tokenizer
  tokenizer = BertTokenizer.from_pretrained(BETO, do_lower_case=True)

  # Specify `MAX_LEN`
  MAX_LEN = returnMaxLength(X_train, X_val, X_test, tokenizer)

  train_dataloader, val_dataloader = createDataloader(X_train, X_val, y_train, y_val, tokenizer, MAX_LEN)

  set_seed(42)    # Set seed for reproducibility
  bert_classifier, optimizer, scheduler = initialize_model(device, train_dataloader, epochs=2)
  train(device, optimizer, scheduler, bert_classifier, train_dataloader, val_dataloader, epochs=2, evaluation=True)

  full_train_dataloader = createFullDataloader(X_train, X_val, y_train, y_val, tokenizer, MAX_LEN)

  # Train the Bert Classifier on the entire training data
  set_seed(42)
  bert_classifier, optimizer, scheduler = initialize_model(device, full_train_dataloader, epochs=2)
  train(device, optimizer, scheduler, bert_classifier, full_train_dataloader, epochs=2)

  if(save_model is True):
    bert_classifier.bert.save_pretrained(save_directory)

  return bert_classifier

df_train = pd.read_csv(train_dataset, sep='\t')
df_test = pd.read_csv(test_dataset, sep='\t')

beto_classifier = fineTunning(df_train, df_test, save_model=save_flag)