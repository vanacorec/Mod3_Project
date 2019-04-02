import json
import pandas as pd 




def get_all_df(jsonl_files_list):
  """ Gets a tweets and users dataframe for data in all jsonl files and writes them to csv files
  Parameters:
    json1_files_list (iterable): list of one or more jsonl files
  Returns:
    all_tweets_df: dataframe of all tweets data
    all_users_df: dataframe of all users data 
  """
  all_tweets_df_list = []
  all_users_df_list = []
  for jsonl_file in jsonl_files_list:
    tweets_df, users_df = get_user_tweet_dfs(jsonl_file)
    all_tweets_df_list.append(tweets_df)
    all_users_df_list.append(users_df)
  all_tweets_df = pd.concat(all_tweets_df_list)
  all_users_df = pd.concat(all_users_df_list)
  
  
  all_tweets_df.to_csv('all_tweets_db.csv')
  all_users_df.to_csv('all_users_db.csv')
  return all_tweets_df, all_users_df
  


def get_user_tweet_dfs(jsonl_file):
  '''
  Takes in a jsonl file
  Gets tweet and user dictionaries for each line and adds
  them as a new row in a tweets and users dataframe, respectively

  Returns 2 dataframes
  '''

  tweet_dfs=[]
  user_dfs=[]
  
  with open(jsonl_file) as f:

    for line in f:
      
      temp=json.loads(line)
      tweet,user=parse_json(temp)
      
      tweet_dfs.append(pd.DataFrame(tweet,index=[tweet['tweet_id_str']]))
      user_dfs.append(pd.DataFrame(user,index=[user['user_id_str']]))
        
  tweets_df = pd.concat(tweet_dfs)
  users_df = pd.concat(user_dfs)
  
  return tweets_df, users_df


def parse_json(tweet):
  '''
  Takes in a line from the jsonl file
  Parses the line into two dictionaries:
    1. Tweet dictionary with info about the tweet: created date string, hashtags,
    favorite count, text, tweet id, language, retweet count, retweeted, 
    user id, tweet id string, and mentions
    2. User dictionary with info about the user who tweeted: id, screen name, user id string
    followers count, statuses count, language, time zone, verfied, created date, favorites count, 
    friends count, listed count

    Returns 2 dictionaries
  '''
  cleaned={}
  user={}
  
  try:
    cleaned['created_str']=tweet['created_at']
    if tweet['entities']['hashtags']:
      cleaned['hashtags']=[[hashtag['text'] for hashtag in tweet['entities']['hashtags']]]
    else:
      cleaned['hashtags']=[[]]
    cleaned['favorite_count']=tweet['favorite_count']
    cleaned['text']=tweet['full_text']
    cleaned['tweet_id']=tweet['id']
    cleaned['lang']=tweet['lang']
    cleaned['retweet_count']=tweet['retweet_count']
    cleaned['retweeted']=tweet['retweeted']
    cleaned['user_id']=tweet['user']['id']
    cleaned['tweet_id_str']=tweet['id_str'] 
   
    if tweet['entities']['user_mentions']:
      cleaned['mentions']=[[mentions['screen_name'] for mentions in tweet['entities']['user_mentions']]]
    else:
      cleaned['mentions']=[[]]
  except:
    print("tweet exception: ",tweet['id'])
  
  try:
    user['id']=tweet['user']['id']
    user['screen_name'] = tweet['user']['screen_name']
    user['user_id_str'] = tweet['user']['id_str']
    user['followers_count']=tweet['user']['followers_count']
    user['statuses_count']=tweet['user']['statuses_count']
    user['lang']=tweet['user']['lang']
    user['time_zone']=tweet['user']['time_zone']
    user['verified']=tweet['user']['verified']
    user['created_at']=tweet['user']['created_at']
    user['favourites_count']=tweet['user']['favourites_count']
    user['friends_count']=tweet['user']['friends_count']
    user['listed_count']=tweet['user']['listed_count']
  except:
    print("user exception: ", tweet['user']['id'])
   
  
  return cleaned,user


