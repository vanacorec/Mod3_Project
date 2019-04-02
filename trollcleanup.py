import pandas as pd 
import numpy as np 
import re

def to_list(string,quotes):
	""" Converts a string into a list of strings
	
	Parameters:
		string (str): A literal string representation of a list.
		quotes (char): The type of string wrapper of items in the list.

	Returns:
		items ([str]): A list of strings stripped of the wrapping brackets.
	"""
	# Find strings within given quotes
	if quotes == '"':
	    pattern = r"\"(.*?)\""
	else:
		pattern = r"\'(.*?)\'"
    # Checks for null values, returns empty list
    if type(string) != str:
        return []
    # Finds all strings within the brackets, returned as a list
    items = re.findall(pattern, string)
    return items

def to_list_count(lst):
	'''Count the elements of a list'''
    return len(lst)

def add_counts(df):
	""" Count the number of hashtags and mentions for each observation in a DataFrame.

	Parameters:
		df (DataFrame): Pandas DF where each observation is a tweet and which contains
			columns with a list of hashtags and with a list of mentions

	Returns:
		df (DataFrame): Returns Pandas DF with 'hashtags' and 'mentions' column changed 
			from a str to a list of str. Adds columns for the count of each of the str 
			in those columns.
	"""

    df["hashtags"] = df["hashtags"].apply(to_list)
    df["hashtags_count"] = df["hashtags"].apply(to_list_count)
    
    df["mentions"] = df["mentions"].apply(to_list)
    df["mentions_count"] = df["mentions"].apply(to_list_count)
    return df

def hashtag_counter(series):
	""" Return a count of each unique item in a list of lists.

	Parameters:
		series (list): List where each element is a list of strings.

	Returns:
		counter (Counter): Count of each unique string element in the series.
	
	"""

	from collections import Counter

	counter=Counter()
	
	# Loop through each row in the series and raise the count of each tag in the row
	for row in series:
	    for tag in row:
	        counter[tag.lower()] +=1
    return counter

def clean_tweets_df(tweets_df, target_val):
	""" Take in a DataFrame of tweets and a classification value and return it cleaned.

	Parameters:
		tweets_df (DataFrame): Pandas DataFrame where each row corresponds to a tweet.
			Null values are removed. Engineered features are added. Non-english tweets
			are dropped as are duplicate tweets. Classification column is added and
			then tweet content is cleaned.
		target_val (integer 0 or 1): Classification value where 1 corresponds to a 
			tweet that has been labeled as a Troll and 0 corresponds to a normal 
			tweet.

	Returns:
		tweets_df (DataFrame): Pandas DataFrame with data cleaned and ready for 
			joining.
	"""
	if values:
		tweets_df=tweets_df.drop(columns = ['posted','expanded_urls', 'source', 'retweeted_status_id', 'in_reply_to_status_id'],axis=1)
	else:
		tweets_df=tweets_df.drop(columns = ["Unnamed: 0", "tweet_id_str", "tweet_id", "user_id", "user_id_str"],axis=1)

    tweets_df = tweets_df.drop(tweets_df[tweets_df.tweet_id.isnull()].index)
    tweets_df = tweets_df.drop(tweets_df[tweets_df.text.isnull()].index)
    tweets_df = fix_tweet_id_str(tweets_df)    
    tweets_df = add_counts(tweets_df)
    tweets_df = add_date_time_col(tweets_df)
    tweets_df = remove_non_en(tweets_df)
    tweets_df = remove_dup_tweet_ids(tweets_df)
    tweets_df = add_target_col(tweets_df, target_val)
    tweets_df["retweeted"] = tweets_df['text'].apply(is_rt)
    tweets_df['text'] = tweets_df['text'].apply(strip_tweets)
    
    return tweets_df

def remove_dup_tweet_ids(df):
	'''Print the size of a DataFrame and remove duplicate values by tweet id.'''
    print(len(df))
    df = df[~df.tweet_id_str.duplicated(keep='first')]
    print(len(df))
    return df

def add_target_col(df, val):
	'''Append column with classification label.'''
    df['target'] = val
    return df
def remove_non_en(df):
	'''Return only entries in English.'''
    df = df[df['lang'] == 'en']
    return df
def add_date_time_col(df):
	'''Append a column of the tweet time as a DateTime.'''
    df['date_time'] = pd.to_datetime(df['created_str'])
    return df
def fix_tweet_id_str(df):
	'''Return the index as the tweet_id as a str.'''
    return df.drop("tweet_id_str", axis = 1).rename(columns = {"Unnamed: 0" : "tweet_id_str"})
def drop_null_tweet_ids(df):
	'''Drop all entries with a null tweet_id.'''
    return df.drop(df[df.tweet_id.isnull()].index)

def strip_tweets(tweet):
	'''Process tweet text to remove retweets, mentions,links and hashtags.'''
    retweet = r'RT:? ?@\w+:?'
    tweet= re.sub(retweet,'',tweet)
    mention = r'@\w+'
    tweet= re.sub(mention,'',tweet)
    links = r'^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$'
    tweet= re.sub(links,'',tweet)
    tweet_links = r'https:\/\/t\.co\/\w+|http:\/\/t\.co\/\w+'
    tweet=re.sub(tweet_links,'',tweet)
    hashtag = r'#\w+'
    tweet= re.sub(hashtag,'',tweet)
    return tweet
def is_rt(string):
	'''Determine whether the tweet is a retweet, returned as a 0 for no and 1 for yes.'''
    retweet = r'RT:? ?@\w+:?'
    if re.findall(retweet, string):
        return 1
    else:
        return 0
