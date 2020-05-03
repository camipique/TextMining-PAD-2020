# -*- coding: utf-8 -*-
"""
PAD Project - Text Mining

@author: Carlos Quendera 49946
@author: David Pais 50220
"""

# Part 1 - Extracting relevant words

import re, os
from nltk import FreqDist
from nltk.util import ngrams

CORPUS_FOLDER_PATH = "corpus2mwTest/"

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

omegaMinus1 = {
        1: {},
        2: {},
        3: {},
        4: {},
        5: {},
        6: {}
}


omegaPlus1 = {
        0: {},
        1: {},
        2: {},
        3: {},
        4: {},
        5: {}
}




word_count = len(words)

def n_gram_prob(n_gram): # [dictionary entry][element inside the dictionary entry]
    return n_grams_freq[len(n_gram) - 1][n_gram] / word_count

def glue(n_gram):
    
    f = 0 #frequency

    if len(n_gram) == 1:
        return n_gram_prob(n_gram)

    for i in range(1, len(n_gram)): # starting in 1 because :1 goes till the start index, so starts in 0. i: starts in the index (1) till the index
        f += n_gram_prob( n_gram[:i] ) * n_gram_prob( n_gram[i:] )


    f = f / ( len(n_gram) - 1 ) # formula of F 

    
    return n_gram_prob(n_gram) ** 2 / f # formula of glue of a n-gram with n > 2 (it can also apply to n = 2)
    
mwu = set([])
# main method

for i in range(6, 0, -1): # starts in 1 till 6 (inclusive)
    for n_gram in n_grams[i]: # in dictionary entry, and inside that entry glues[i][ngram] calculate the glues
        
        n_gram_glue = glues[i][n_gram] = glue(n_gram)
        
        omegaMinus1[i][n_gram] = [glue(n_gram[1:])]  
        omegaMinus1[i][n_gram].append( glue(n_gram[ :i]) )
        
        
        try: # n_gram[1:] in omegaPlus1[i-1]:
            omegaPlus1[i-1][n_gram[1:]].append(glue(n_gram))

        except KeyError:
            omegaPlus1[i-1][n_gram[1:]] = [glue(n_gram)]
        
        try: # n_gram[:i] in omegaPlus1[i-1]:
            omegaPlus1[i-1][n_gram[:i]].append(glue(n_gram))
        
        except KeyError:
            omegaPlus1[i-1][n_gram[:i]] = [glue(n_gram)]
             


for i in range(1, 6):
    for n_gram in n_grams[i]:
        
        if n_grams_freq[i][n_gram] >= 2:    
            n_gram_glue = glues[i][n_gram]
            
            if i == 1:
                if n_gram_glue > max(omegaPlus1[i][n_gram]):
                    mwu.add((n_gram_glue, " ".join(n_gram)))
            else:
                x = max(omegaMinus1[i][n_gram])
                y = max(omegaPlus1[i][n_gram])
                
                if n_gram_glue > (x + y) / 2:
                    mwu.add((n_gram_glue, " ".join(n_gram)))
                
            
        
with open("mwu.txt", "w", encoding="utf-8") as file:
    for relevant_expression in mwu:
        file.write(str(relevant_expression) + "\n")

        