# -*- coding: utf-8 -*-
"""
PAD Project - Text Mining

@author: Carlos Quendera 49946
@author: David Pais 50220
"""

# Part 1 - Extracting relevant words


import re, os
from nltk import FreqDist

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

split_text_str = ""
#split_text_list = []



# with - execute the operations as pairs
for file_name in os.listdir(CORPUS_FOLDER_PATH):  
    with open(CORPUS_FOLDER_PATH + file_name, "r", encoding="utf8") as f:
        text = f.read()
        
        # find the regex defined in text
        text_list = re.findall(regex, text)
         
        split_text_str += " ".join(text_list)
#       split_text_list.extend(text_list)

# still need to verify this last sentence, only giving frequency about letters
word_freq = FreqDist(split_text_str)
