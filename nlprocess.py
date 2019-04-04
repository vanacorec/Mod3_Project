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



'''
Combined list of: generic stopwords downloaded from nltk, generic punctuation, stopwords specific to our project
and punctuation specific to our project - all to be removed during the tokenizing step of preprocessing for NLP
'''
stopwords_list=stopwords.words('english') +list(string.punctuation)
stopwords_list += ["'",'"','...','``','…','’','‘','“',"''",'""','”','”',"'s'",'\'s','n\'t','\'m','\'re','amp','https']

def process_tweet(tweet):
	""" Takes in a string, returns a list words in the string that aren't stopwords

	Parameters:
		tweet (string):  string of text to be tokenized

	Returns:
		stopwords_removed (list): list of all words in tweet, not including stopwords
	"""

	tokens = nltk.word_tokenize(tweet)
	stopwords_removed = [token.lower() for token in tokens if token not in stopwords_list]
	return stopwords_removed

def tokenized(series):
	""" Takes in a series containing strings or lists of strings, and creates a single list of all the words

	Parameters:
		series (series): series of text in the form of strings or lists of string 
	
	Returns:
		tokens (list): list of every word in the series, not including stopwords
	"""
	
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
	""" Take in a dataframe with a column of text strings, count vectorizes the text andt then tfidf transform 

	Parameters:
		text (dataframe): dataframe with a column of strings named "text" 

	Returns:
		cv (CountVectorizer): count vectorized text
		tfidf (TfidfTransformer): TFidfTranformed count vectors  
	"""
	cv = CountVectorizer(stop_words=stopwords_list)
	tfidf = TfidfTransformer(use_idf=IDF)

	cv.fit(text.text)
	dummy_vocab=cv.transform(text.text)
	tfidf.fit(dummy_vocab)
	return cv,tfidf

def data_sampler(features,target, size=.75):
	""" Takes in a dataframe of features and a series of corresponding class labels, splits them into train and test subsets, 
	and undersamples the training subsets to fix the class imbalance 

	Parameters:
		features (dataframe): dataframe containing columns of features
		target (series): corresponding class labels for each row in the features dataframe
		size (int) (optional): portion of data to include in the train subset, default is .75

	Returns:
		X_rus (np.ndarray) : training data (undersampled because of class imbalance)
		X_test (dataframe): testing data
		y_rus (np.ndarray): training target (undersampled because of class imbalance)
		y_test (series):  testing target
	""" 
	X_train,X_test,y_train,y_test=train_test_split(features,target,train_size=size,random_state=19,stratify=target)
	rus = RandomUnderSampler(random_state=19)
	X_rus,y_rus = rus.fit_sample(X_train,y_train)

	return X_rus, X_test, y_rus, y_test


