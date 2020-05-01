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
from nltk.util import everygrams

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


n_grams_freq = dict(FreqDist(n_grams))


key_list = list(n_grams_freq.keys()) 
val_list = list(n_grams_freq.values())

""" Calculate glues """

""" For n == 2 """ ## then we can do a for with n cycle for this (for all n's)

# get size of n-gram
gram_2size = np.size(key_list[0])

# get sentence of n-gram 
gram_2 = key_list[0]

# get singular words
firstW = gram_2[0]
secondW = gram_2[1]

# get singular words frequency in text


""" Calculate omega n - 1 """

# For the 2 gram is just the single words frequency (we would add these values to the omega set - t1 and t2)

#omegaN_minus1 = []

# here we would put a for to append this to the omegaN_minus list
firstWfreq = word_freq[firstW]
#omegaN_minus1.append(firstWfreq)
secondWfreq = word_freq[secondW]
#omegaN_minus1.append(secondWfreq)

#omegaN_plus1 = []

# get 2-gram frequency
gram_2freq = val_list[0]

# get 2-gram glue
glue_2 = (gram_2freq ** 2)  / (firstWfreq  * secondWfreq)





""" For n > 2, in this case n == 3 """

# get size of n-gram (this is just informative)
gram_3size = np.size(key_list[6000])

# get sentence of n-gram 
gram_3 = key_list[6000]

# get 3-gram frequency
gram_3freq = val_list[6000]

# get singular words of the 3-gram
singular_words = []

word_frequencies = []

divisions = []

#division n-grams frequency
frequencySubGrams = []

for i in range(len(gram_3)):
    singular_words.append(gram_3[i])  # this is not needed, maybe use directly e13 or other vector name
# get singular words frequency in text
    word_frequencies.append(word_freq[singular_words[i]])    
#get n-grams division frequencies
    divisions_x = list(gram_3)
    divisions_y = list(gram_3)
    # is always less 1 than the size of the n-gram  
    ## here before we can put an if instead of a for? if has not got to the last element of the array continue
    if i < len(gram_3) - 1:
        print("\t\t\n New Division \t\t")
        x = divisions_x[: - (len(gram_3) - 1 - i)]
        y = divisions_y[-(len(gram_3) - 1 - i):]
        print('x = ' + str(x))
        print('y = ' + str(y)+ '\n')
        for j in x, y: 
            print('j = ' + str(j))
            if(len(j) == 1): #index 863 'The', 'beginning'
                value = word_freq[j[0]]
                frequencySubGrams.append(value)
                print('len 1 \t freq =  ' + str(value) + '\n')
            else:
                value = val_list[key_list.index(tuple(j))]
                print('len > 1 \t freq = ' + str(value) + '\n')
                frequencySubGrams.append(value)
        

probSubGrams = np.asarray(frequencySubGrams) / len(text_list)
#mult = len(tete) / 2
    
# Ta a funcionar fixolas
probNGram = sum(probSubGrams[x] * probSubGrams[y] for x,y in zip(range(0,len(probSubGrams),2), range(1,len(probSubGrams),2) ))

glue_3 = ((gram_3freq / len(text_list)) ** 2 ) /  (  (1 / (len(gram_3) - 1) ) * (probNGram) )

