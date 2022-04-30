import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainerState, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import re
import sys
import os

# Model to use
MODEL_NAME = 'None'

# directories of the datasets to use
train_dataset = 'None'
test_dataset = 'None'

# labels of the dataset that represents each annotated column
labels = ['gender', 'profession', 'ideology_binary', 'ideology_multiclass']

# directory to save the trained model
save_directory = 'None'
save_flag = False

argv = sys.argv
argc = len(argv)
arguments = {"--train_data": "", "--test_data": "", "--model_name": "", "--labels": [], "--save_model_on_directory": ""}
arguments_stack = ["--train_data", "--test_data", "--model_name", "--labels", "--save_model_on_directory"]
mandatories = ["--train_data", "--test_data", "--model_name"]
i = 1
while i < argc:
    arg = argv[i]
    if(arg in arguments_stack):
        i += 1
        ended = False
        while not ended and i < argc:
            if arg == "--labels":
                if re.match("^--", argv[i]):
                    arguments_stack.remove(arg)
                    ended = True
                    i -= 1

                else:
                    arguments[arg].append(argv[i])

            else:
                arguments[arg] = argv[i]
                arguments_stack.remove(arg)
                ended = True

            print(arg + ": " + argv[i])
            i += 1

    else:
        print("Usage: python SentimentTweets.py --train_data <directory> --test_data <directory> --model_name <name_or_path> [--labels <label1 label2 ... labeln>] [--save_model_on_directory <directory>]")
        sys.exit(-1)

for arg in arguments_stack:
    if arg in mandatories:
        print("Usage: python SentimentTweets.py --train_data <directory> --test_data <directory> --model_name <name_or_path> [--labels <label1 label2 ... labeln>] [--save_model_on_directory <directory>]")
        sys.exit(-1)

MODEL_NAME = arguments["--model_name"]
train_dataset = arguments["--train_data"]
test_dataset = arguments["--test_data"]
labels = arguments["--labels"] if arguments["--labels"] != [] else labels
save_directory = arguments["--save_model_on_directory"]
save_flag = True if save_directory != [] else False


# Load the MODEL_NAME tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

# Load the MODEL_NAME model
models = {}
for key in labels:
    if key == 'ideology_multiclass':
        models[key] = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    else:
        models[key] = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# max tensor length
max_length = max_length=models[labels[0]].config.max_position_embeddings
max_length = max_length if max_length < 400 else 400


def process_long_url(url):
    url = re.sub(r'https://[:a-zA-Z_0-9-.&?]*', ' ', url) 
    url = re.sub(r'http://[:a-zA-Z_0-9-.&?]*', ' ', url) 
    url = url.replace('www.','')
    url = url.replace('.com','')
    url = url.replace('/',' ')
    url = url.replace('-',' ')
    return url

def process_urls(text):
    res = re.findall(r'http[:/a-zA-Z_0-9-.&?]*', text)
    for i in res:
        if len(i)<30: # corta  
            text= text.replace(i,' ')

        else:
            text = text.replace(i,process_long_url(i))

    return text

def remove_users(text):
    return re.sub(r'@[a-zA-Z_0-9-]*', ' ', text)


def remove_numbers(text):
    return re.sub(r'[0123456789]+', ' ', text)


def transform_hastag(hashtag):
    newHastag=''
    for c in hashtag:
        if c.isupper() == True:
            newHastag=newHastag+' '+c

        else:
            newHastag=newHastag+c

    newHastag =newHastag.strip()
    newHastag=newHastag.replace('#',' ')
    return newHastag

def process_hashtags(text):
    res = re.findall(r'#[a-zA-Z_0-9]*', text)
    for i in res:
        text = text.replace(i, transform_hastag(i))

    return text


def process_tweet(text, lang = None):
    text = remove_users(text)
    text = process_urls(text)

    text = process_hashtags(text)
    text = text.replace('/',' ')
    text = remove_numbers(text)
    #Remove multiple whitespaces (I think that in the tokenization stage this becomes irrelevant but I've added just in case).
    text = ' '.join(text.split())
    #Option to remove stopwords, I've removed from the stopword list no and not for cases like 'no deberías salir de la cocina' (although, it keeps soundig misogynistic)
    #text = remove_stopwords(text, lang)

    
    return text


def text_preprocessing(group):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    result = []

    for text in group:
        text = process_tweet(text)
        # Remove '@name'
        #text = re.sub(r'(@.*?)[\s]', ' ', text)

        # Replace '&amp;' with '&'
        #text = re.sub(r'&amp;', '&', text)

        # Remove trailing whitespace
        #text = re.sub(r'\s+', ' ', text).strip()

        # Remove punctuation
        #text = re.sub(r'[^\w\s]', '', text)

        result.append(text)

    return result


def create_datasets(*args):
    result = []
    for arg in args:
        result.append(Dataset.from_pandas(arg))
    
    return result


# Create a function to tokenize a set of texts
def preprocessing_tokenizer(input):
    return tokenizer(text_preprocessing(input["text"]), truncation=True, padding=True, max_length=max_length)


def preprocess(*args):
    result = []
    for arg in args:
        aux = arg.map(preprocessing_tokenizer, batched=True,load_from_cache_file=False)
        aux.set_format(type='torch', columns=['input_ids', 'label', 'attention_mask'])
        result.append(aux)

    return result


def compute_metrics(eval_preds):
    logits, y_true = eval_preds
    y_pred = np.argmax(logits, axis=-1)
    return { 'precision': precision_score(y_true, y_pred, average='macro'), 'recall': recall_score(y_true, y_pred, average='macro'), 'accuracy': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, average='macro') }


def my_trainer(preprocessed_train, preprocessed_eval, model, save_model=False):
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
        eval_dataset=preprocessed_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer


def change_label_datasets(df_train, df_eval):
    ds_group = {}
    for key in labels:
        new_df_train = df_train[[key, 'text']]
        new_df_eval = df_eval[[key, 'text']]
        new_df_train.rename(columns={key: 'label'}, inplace=True)
        new_df_eval.rename(columns={key: 'label'}, inplace=True)
        ds_group[key] = create_datasets(new_df_train, new_df_eval)

    return ds_group


def preprocess_for_labels(ds_group):
    preprocessed_group = {}
    for key in labels:
        preprocessed_group[key] = preprocess(ds_group[key][0], ds_group[key][1])

    return preprocessed_group


def output_file(df_output):
    df_output['gender'][df_output['gender'] == 0] = 'female'
    df_output['gender'][df_output['gender'] == 1] = 'male'

    df_output['profession'][df_output['profession'] == 0] = 'journalist'
    df_output['profession'][df_output['profession'] == 1] = 'politician'

    df_output['ideology_binary'][df_output['ideology_binary'] == 0] = 'left'
    df_output['ideology_binary'][df_output['ideology_binary'] == 1] = 'right'

    df_output['ideology_multiclass'][df_output['ideology_multiclass'] == 0] = 'left'
    df_output['ideology_multiclass'][df_output['ideology_multiclass'] == 1] = 'moderate_left'
    df_output['ideology_multiclass'][df_output['ideology_multiclass'] == 2] = 'moderate_right'
    df_output['ideology_multiclass'][df_output['ideology_multiclass'] == 3] = 'right'

    return df_output


def fineTunning(train_data, test_data, save_model = False):
    df_train, df_eval = train_test_split(train_data, test_size=0.15, random_state=2022)

    ds_group = change_label_datasets(df_train, df_eval)
    ds_test = create_datasets(test_data)

    ds_test = ds_test[0].map(preprocessing_tokenizer, batched=True,load_from_cache_file=False)
    ds_test.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    preprocessed_group = preprocess_for_labels(ds_group)

    trainers = {}
    f1_group = {}
    for key in labels:
        trainer = my_trainer(preprocessed_group[key][0], preprocessed_group[key][1], models[key], save_model=save_model)
        trainers[key] = trainer
    
        metrics = trainer.train()
        print(metrics)
  
        if(save_model):
            models[key].save_pretrained(save_directory + key)
            tokenizer.save_pretrained(save_directory + key)

        result = trainer.evaluate(preprocessed_group[key][1])
        f1_group[key] = result['eval_f1']

    final_f1 = 0
    for key in labels:
        final_f1 += f1_group[key]
        print('f1_score for ' + key + ': ' + str(f1_group[key]))

    print('final f1 score: ' + str(final_f1/len(f1_group)))

    predictions = {}
    predictions['user'] = test_data['user'].values
    for key in labels:
        output = trainers[key].predict(Dataset.from_pandas(df_test))
        predictions[key] = np.argmax(output[0], axis=-1)

    df_output = pd.DataFrame(predictions)
    #new_df_output = df_output.groupby('user')
    #df_output = new_df_output.agg(lambda x: np.argmax(np.unique(x, return_counts=True)[1]))
    #df_output = output_file(df_output)

    if(save_model):
        df_output.to_csv(save_directory + "results registry/result.csv")


df_train = pd.read_csv(train_dataset, sep='\t')
df_test = pd.read_csv(test_dataset, sep='\t')

trainer = fineTunning(df_train, df_test, save_model=save_flag)