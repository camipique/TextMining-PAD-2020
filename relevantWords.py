# -*- coding: utf-8 -*-
"""
PAD Project - Text Mining

@author: Carlos Quendera 49946
@author: David Pais 50220
"""

# Part 1 - Extracting relevant words

import re, os
import numpy as np
from nltk import FreqDist
from nltk.util import everygrams, ngrams

CORPUS_FOLDER_PATH = "corpus2mw/"

def find_n_grams(text):
    words = list(ngrams(text, 1))
    bigrams = list(ngrams(text, 2))
    trigrams = list(ngrams(text, 3))
    fourgrams = list(ngrams(text, 4))
    fivegrams = list(ngrams(text, 5))
    sixgrams = list(ngrams(text, 6))
    sevengrams = list(ngrams(text, 7))

    return words, bigrams, trigrams, fourgrams, fivegrams, sixgrams, sevengrams


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

n_grams_freq = dict(FreqDist(n_grams))


#key_list = list(n_grams_freq.keys()) 
#val_list = list(n_grams_freq.values())
#
#""" Calculate glues """
#
#""" For n == 2 """ ## then we can do a for with n cycle for this (for all n's)
#
## get size of n-gram
#e1 = np.size(key_list[0])
#
## get sentence of n-gram 
#e1 = key_list[0]
#
## get singular words
#eheh = e1[0]
#eheh2 = e1[1]
#
## get singular words frequency in text
#
#t1 = word_freq[eheh]
#t2 = word_freq[eheh2]
#
## get 2-gram frequency
#e1_together = val_list[0]
#
## get 2-gram glue
#glue_2 = (e1_together ** 2)  / (t1 * t2)
#
#
#""" For n > 2, in this case n == 3 """
#
## get size of n-gram
#e13 = np.size(key_list[6000])
#
## get sentence of n-gram 
#e13 = key_list[6000]
#
## get 3-gram frequency
#e13_together = val_list[6000]
#
## get singular words
#
#singular_words = []
#word_frequencies = []
#divisions = []
#
##division n-grams frequency
#ti = []
#
#for i in range(len(e13)):
#    singular_words.append(e13[i])  # this is not needed, maybe use directly e13 or other vector name
## get singular words frequency in text
#    word_frequencies.append(word_freq[singular_words[i]])    
##get n-grams division frequencies
#    divisions_x = list(e13)
#    divisions_y = list(e13)
#    # is always less 1 than the size of the n-gram  
#    ## here before we can put an if instead of a for? if has not got to the last element of the array continue
#    if i < len(e13) - 1:
#        print("\t\t\n New Division \t\t")
#        x = divisions_x[: - (len(e13) - 1 - i)]
#        y = divisions_y[-(len(e13) - 1 - i):]
#        print('x = ' + str(x))
#        print('y = ' + str(y)+ '\n')
#        for j in x, y: 
#            print('j = ' + str(j))
#            if(len(j) == 1): #index 863 'The', 'beginning'
#                value = word_freq[j[0]]
#                ti.append(value)
#                print('len 1 \t freq =  ' + str(value) + '\n')
#            else:
#                value = val_list[key_list.index(tuple(j))]
#                print('len > 1 \t freq = ' + str(value) + '\n')
#                ti.append(value)
#        
#
#tete = np.asarray(ti) / len(text_list)
#mult = len(tete) / 2
#    
## Ta a funcionar fixolas
#yoyo = sum(tete[x] * tete[y] for x,y in zip(range(0,len(tete),2), range(1,len(tete),2) ))
#
#
#glue_3 = ((e13_together/ len(text_list)) ** 2 ) /  (  (1 / (len(e13) - 1) ) * (yoyo) )

words, bigrams, trigrams, fourgrams, fivegrams, sixgrams, sevengrams = find_n_grams(text_split_list)

n_grams = {
        0: words,
        1: bigrams,
        2: trigrams,
        3: fourgrams,
        4: fivegrams,
        5: sixgrams,
        6: sevengrams
}

n_grams_freq = {
        0: FreqDist(words),
        1: FreqDist(bigrams),
        2: FreqDist(trigrams),
        3: FreqDist(fourgrams),
        4: FreqDist(fivegrams),
        5: FreqDist(sixgrams),
        6: FreqDist(sevengrams)
}

glues = {
        1: {},
        2: {},
        3: {},
        4: {},
        5: {},
        6: {}
}

word_count = len(words)

def n_gram_prob(n_gram):
    return n_grams_freq[len(n_gram) - 1][n_gram] / word_count

def glue(n_gram):
    
    f = 0
    
    for i in range(1, len(n_gram)):
        f += n_gram_prob( n_gram[:i] ) * n_gram_prob( n_gram[i:] )
        
    f = f / ( len(n_gram) - 1 )
    
    return n_gram_prob(n_gram) ** 2 / f
    
for i in range(1, 7):
    for n_gram in n_grams[i]:
        glues[i][n_gram] = glue(n_gram)


