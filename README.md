# beto-covid-sentiment-analysis
Research project of Sentiment Analysis based on [this code](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/) and using [this BETO model](https://github.com/dccuchile/beto).

_Note_: to gain access to the dataset, please, contact with [SerPablo (Pablo Calleja)](https://github.com/SerPablo)

## Description of the Scripts
- ### _RetrieveTweets.py_:
  This script creates a dataset of tweets based on the keywords given and the number of tweets to be retrieved.
  - How to use it:
    ```
    python RetrieveTweets.py --search <keyword> --amount <number_of_tweets> --save_tweets_on_directory <directory> --twitter_token <your_token>
    ```
    - _--search_: parameter to search the keyword or keywords. If more than one keyword is going to be used then you need to quote them ('your keywords').
    - _--amount_: parameter to specify the number of tweets to retrieve (be aware that if considerable amount of tweets are going to be solicited, you may exceed Twitter's limitation of queries per minute).
    - _--save_tweets_on_directory_: parameter to specify where to store the tweets retrieved.
    - _--twitter_token_: parameter to specify your Twitter developer access token.
- ### _Preprocesing.py_:
  This script cleans the previous dataset obtained with _RetrieveTweets.py_ for it to be suitable to be used on _SentimentTweets.py_.
  - How to use it:
    ```
    python Preprocesing.py --dataset <directory> --save_directory <directory> [--merge [name][description]]
    ```
    - _--dataset_: parameter to specify the dataset to be cleaned.
    - _--save_directory_: parameter to specify where to store the cleaned dataset.
    - _--merge_: parameter to specify which columns will be merged with the _tweet_text_ column. It can be both (_name_ and _description_), just one of them or none.
- ### _CustomModel.py_:
  This script contains the code necessary to build the model class. It is imported in the script _SentimentTweets.py_ for it to be used as the model for the fine tuning task. It recieves the name of the BETO based model to be used.
- ### _SentimentTweets.py_:
  This script contains all the code necessary to do the fine tuning task. Recieves the train and test datasets to fine tune the model. It can save the model after fine tuned and gives an output with the results of the training.
  - How to use it:
    ```
    python SentimentTweets.py --train_data <directory> --test_data <directory> --model_name <name_or_path> [--save_model_on_directory <directory>]
    ```
    - _--train_data_: parameter to specify the train dataset to be used.
    - _--test_data_: parameter to specify the test data to be used.
    - _--model_name_: parameter to specify the name of the model (from hugging face) to be used.
    - _--save_model_on_directory_: parameter to specify where to store the fine tuned model.
