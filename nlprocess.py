import re
import nltk
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from wordcloud import WordCloud
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
nltk.download('punkt')
	
stopwords_list=stopwords.words('english') +list(string.punctuation)
stopwords_list += ["'",'"','...','``','…','’','‘','“',"''",'""','”','”',"'s'",'\'s','n\'t','\'m','\'re','amp','https']

def process_tweet(tweet):
	tokens = nltk.word_tokenize(tweet)
	stopwords_removed = [token.lower() for token in tokens if token not in stopwords_list]
	return stopwords_removed

def tokenized(series):
	corpus = ' '.join([tweet.lower() if type(tweet)==str else ' '.join([tag.lower() for tag in tweet]) for tweet in series])
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
	frequencies = FreqDist(tokenized(series))
	return frequencies.most_common(top)

def create_wordcloud(series, *top):
	""" Take in a list of lists and create a WordCloud visualization for those terms.

	Parameters:
			series (iterable): A list of lists containing strings.
	Returns:
		None: The ouput is a visualization of the strings in series in terms of the
			frequency of their occurrence.

	"""
	# if top[0]:
	# 	series=wordfrequency(series,top[0])
	vocab = tokenized(series)
	if not top[0]:
		top[0]=200
	cloud=WordCloud(max_words=top[0]).generate(' '.join([word for word in vocab]))
	plt.imshow(cloud,interpolation='bilinear')
	plt.plot(figsize = (8,4))
	plt.axis('off')
	plt.show();

def text_process(text,IDF):
	cv = CountVectorizer(stop_words=stopwords_list)
	tfidf = TfidfTransformer(use_idf=IDF)

	cv.fit(text.text)
	dummy_vocab=cv.transform(text.text)
	tfidf.fit(dummy_vocab)
	return cv,tfidf

def data_sampler(features,target, size=.75):
	X_train,X_test,y_train,y_test=train_test_split(features,target,train_size=size,random_state=19,stratify=target)
	rus = RandomUnderSampler(random_state=19)
	X_rus,y_rus = rus.fit_sample(X_train,y_train)

	return X_rus, X_test, y_rus, y_test