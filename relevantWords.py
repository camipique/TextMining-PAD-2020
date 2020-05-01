# -*- coding: utf-8 -*-
"""
PAD Project - Text Mining

@author: Carlos Quendera 49946
@author: David Pais 50220
"""

# Part 1 - Extracting relevant words

import re, os
from nltk import FreqDist
from nltk.util import ngrams, everygrams

CORPUS_FOLDER_PATH = "corpus2mwTest/"

# Find a set of word characters ([\w'’-])     + means that has at least one ocurrence or more of words followed or not by '’-
# | means or (word characters or punctuation)    
# where the punctuation is [      ] within this set
# [ ; : ! ? < > & ( )  \[    \]   to interpret these not as metacharacters but as [  ] characters itself  
# [ ; : ! ? < > & ( )  \[  \]   \" to not interpret " has a close sign
# [ ; : ! ? < > & ( )  \[  \]  \"  \. , = / \\ (to not interpret \ as an escaoe signal)]
# Not adding spaces on ' and - when they are attached to words
# And also not substituting isolated '’- with white spaces 

regex = re.compile("[\w'’-]+|[;:!?<>&\(\)\[\]\"\.,=/\\\^\$\*\+\|\{\}]|[\S'’-]+")

#regex = re.compile("([\w-]+\'|[\w-]+\’|[^-\w])")

text_split_str = ""
text_split_list = []


# with - execute the operations as pairs
for file_name in os.listdir(CORPUS_FOLDER_PATH):  
    
    with open(CORPUS_FOLDER_PATH + file_name, "r", encoding="utf8") as f:
        text = f.read()
        # find the regex defined in text
        text_list = re.findall(regex, text)
         
        text_split_str += " ".join(text_list)
        text_split_list.extend(text_list)


# We can use the text_split_list instead of transforming text_split_str as a list
word_freq = FreqDist(text_split_list)

n_grams = list(everygrams(text_split_list,min_len=2, max_len=7))


