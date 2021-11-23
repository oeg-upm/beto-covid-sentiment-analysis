import requests
import pandas as pd
import sys
import os

keyword = 'None'
amount = 0
save_directory = 'None'
BEARER_TOKEN = "None"


argv = sys.argv
argc = len(argv)
if(argc != 9):
  print("Usage: python RetrieveTweets.py --search <keyword> --amount <number_of_tweets> --save_tweets_on_directory <directory> --twitter_token <your_token>")
  sys.exit(-1)
 
else:
  keyword = argv[2]
  amount = int(argv[4])
  save_directory = argv[6]
  BEARER_TOKEN = argv[8]
  URL = "https://api.twitter.com/1.1/search/tweets.json"
  headers = {"Authorization": "Bearer {}".format(BEARER_TOKEN)}

def parameters(keyword, nTweets):
  params = {'q': keyword,
            'lang': 'es',
            'count': nTweets,
            #'until': date, # YYYY-MM-DD
            'tweet_mode' : 'extended'}

  return params

def nextPage(keyword, nTweets, max_id):
  params = {'q': keyword,
            'lang': 'es',
            'count': nTweets,
            #'until': date,  # YYYY-MM-DD
            'tweet_mode' : 'extended',
            'max_id' : max_id}

  return params

def getTweets(keyword, nTweets):
  n = int(nTweets / 100)
  remainder = nTweets % 100
  if(n > 0):
    response = requests.request("GET", URL, headers = headers, params = parameters(keyword + ' -filter:retweets', 100))
    print("Endpoint Response Code: " + str(response.status_code))
    json_response = response.json()
    statuses = json_response['statuses']

    for i in range(n-1):
      nextId = json_response['search_metadata']['next_results'][8:27]
      response = requests.request("GET", URL, headers = headers, params = nextPage(keyword + ' -filter:retweets', 100, nextId))
      print("Endpoint Response Code: " + str(response.status_code))
      json_response = response.json()
      statuses = statuses + json_response['statuses']

    if(remainder > 0):
      nextId = json_response['search_metadata']['next_results'][8:27]
      response = requests.request("GET", URL, headers = headers, params = nextPage(keyword + ' -filter:retweets', remainder, nextId))
      print("Endpoint Response Code: " + str(response.status_code))
      json_response = response.json()
      statuses = statuses + json_response['statuses']

  else:
    response = requests.request("GET", URL, headers = headers, params = parameters(keyword + ' -filter:retweets', remainder))
    print("Endpoint Response Code: " + str(response.status_code))
    json_response = response.json()
    statuses = json_response['statuses']

  tweets = pd.DataFrame(statuses)
  df0 = pd.DataFrame({'description' : [i['description'] for i in tweets['user']]})
  df1 = pd.DataFrame({'name' : [i['name'] for i in tweets['user']]})
  tweets = pd.concat([tweets[['created_at', 'id_str', 'full_text', 'retweet_count', 'favorite_count']], df0], axis=1)
  tweets = pd.concat([tweets, df1], axis=1)
  tweets = tweets.rename(columns={'id_str':'id', 'full_text':'tweet_text'})
  return tweets[['created_at', 'id', 'name', 'description', 'tweet_text', 'retweet_count', 'favorite_count']]

# main body
tweets = getTweets(keyword, amount)
output_file = os.path.join(save_directory, 'twitter_dataset.csv')
tweets.to_csv(output_file)