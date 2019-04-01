import pandas as pd 
import numpy as np 

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

def hashtag_counter(series):
	from collections import Counter

	counter=Counter()
	
	for row in series:
	    for tag in row:
	        counter[tag] +=1
    return counter

    