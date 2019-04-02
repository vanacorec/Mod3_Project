import re
import nltk
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from wordcloud import WordCloud
from PIL import Image
	
stopwords_list=stopwords.words('english') +list(string.punctuation)
stopwords_list += ["'",'"','...','``','…','’','‘','“',"''",'""','”','”',"'s'",'\'s','n\'t','\'m','\'re','amp','https']

def process_tweet(tweet):
	tokens = nltk.word_tokenize(tweet)
	stopwords_removed = [token.lower() for token in tokens if token not in stopwords_list]
	return stopwords_removed

def tokenize(series):
	corpus = ' '.join([tweet.lower() for tweet in series])
	tokens = process_tweet(corpus)
	return tokens

def wordfrequency(series, top):
	""" Returns the frequency of words in a list of strings.
	Parameters:
		series (iterable): List of strings to be combined and analyzed
		top (int): The number of top words to return.
	Returns:
		list (tuples): List of word and value pairs for the top words in the series.
	"""
	frequencies = FreqDist(tokenize(series))
	return frequencies.most_common(top)

def create_wordcloud(series, *top):
	""" Take in a list of lists and create a WordCloud visualization for those terms.

	Parameters:
			series (iterable): A list of lists containing strings.
	Returns:
		None: The ouput is a visualization of the strings in series in terms of the
			frequency of their occurrence.

	"""
	if top[0]:
		series=wordfrequency(series,top[0])
	cloud=WordCloud().generate(' '.join([word for word in word_list for word_list in series]))
	plt.imshow(cloud,interpolation='bilinear')
	plt.plot(figsize = (8,4))
	plt.axis('off')
	plt.show();
