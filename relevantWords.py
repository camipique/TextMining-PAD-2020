# -*- coding: utf-8 -*-
"""
PAD Project - Text Mining

@author: Carlos Quendera 49946
@author: David Pais 50220
"""

# Part 1 - Extracting relevant words

import re, os, math
from nltk import FreqDist
from nltk.util import ngrams

CORPUS_FOLDER_PATH = "corpus2mw/"
COHESION_MEASURE = "logLike" # change here to use a different cohesion measure

def find_n_grams(text):
    words = list(ngrams(text, 1))
    bigrams = list(ngrams(text, 2))
    trigrams = list(ngrams(text, 3))
    fourgrams = list(ngrams(text, 4))
    fivegrams = list(ngrams(text, 5))
    sixgrams = list(ngrams(text, 6))
    sevengrams = list(ngrams(text, 7))

    return words, bigrams, trigrams, fourgrams, fivegrams, sixgrams, sevengrams

def n_gram_prob(n_gram): # [dictionary entry][element inside the dictionary entry]
    return n_grams_freq[len(n_gram) - 1][n_gram] / word_count

def glue(n_gram):
    f = cohesion(n_gram, 'glue')
    return n_gram_prob(n_gram) ** 2 / f # formula of glue (scp) of a n-gram with n > 2 (it can also apply to n = 2)
  
def dice(n_gram):
    f = cohesion(n_gram, 'dice')
    return (word_count * n_gram_prob(n_gram) * 2) / f # formula of dice of a n-gram with n > 2 (it can also apply to n = 2)
  
def mi(n_gram):
    f = cohesion(n_gram, 'mi')
    return math.log(n_gram_prob(n_gram) / f) # formula of mi of a n-gram with n > 2 (it can also apply to n = 2)

def phi(n_gram):
    avq, avd = cohesion(n_gram, 'phi')
    return (( (word_count ** 2 * n_gram_prob(n_gram)) - avq) ** 2) / avd # formula of phi of a n-gram with n > 2 (it can also apply to n = 2)

def log_l(p, k, m): # auxiliar function for logLike cohesion measure
        return k * math.log(p) + (m - k) * math.log(1-p)

def logLike(n_gram):
    left_subgram, right_subgram =  cohesion(n_gram, 'logLike')
    kf1 = word_count * n_gram_prob(n_gram)
    kf2 = left_subgram - kf1
    nf1 = right_subgram
    nf2 = word_count - nf1
    pf1 = kf1/nf1 
    pf2 = kf2/nf2
    pf = left_subgram/ word_count

    # formula of logLike of a n-gram with n > 2 (it can also apply to n = 2)
    if( (pf1 > 0 and pf1 < 1) and  (pf2 > 0 and pf2 < 1) and (pf > 0 and pf < 1)): # < 1 because of math.log(1-p) of log_l
       return 2 * ( log_l (pf1, kf1, nf1) + log_l (pf2, kf2, nf2) - log_l (pf, kf1, nf1) - log_l(pf, kf2, nf2))
   
    else:
        return -math.inf # to avoid being bigger than other omega plus 1 values that might be smaller, ln domain goes from [-oo, oo]

def cohesion (n_gram, measure):
    f = 0 # frequency
        
    avq = 0 
    avd = 0
    
    avx = 0
    avy = 0
    
    n_gramSize = len(n_gram)
        
    for i in range(1,  n_gramSize ): # starting in 1 because :1 goes till the start index, so starts in 0. i: starts in the index (1) till the index
        left_subgram = n_gram_prob(n_gram[:i])
        right_subgram = n_gram_prob(n_gram[i:])
        
        if(not (measure == 'dice')):
            if(measure == 'phi'):
                
                left_subgram = left_subgram * word_count
                right_subgram = right_subgram * word_count
                avq += left_subgram * right_subgram 
                avd += (left_subgram * right_subgram) * (word_count - left_subgram) * (word_count - right_subgram)            
                continue
                
            if(measure == 'logLike'):
                
                avx += word_count * left_subgram
                avy += word_count * right_subgram               
                continue
        
            else:                    
                f += left_subgram * right_subgram
                continue
        
        else:
            f += word_count * ( left_subgram + right_subgram)

    f = f / (  n_gramSize  - 1 ) # formula of F 
    
    avq = avq / (  n_gramSize  - 1 )
    avd = avd / (  n_gramSize  - 1 )

    avx = avx / (  n_gramSize  - 1 )
    avy = avy / (  n_gramSize  - 1 )
 
    if(measure == 'phi'):
        return avq, avd
    
    if(measure == 'logLike'):
        return avx,avy

    if(measure == 'glue' or measure == 'dice' or measure == 'mi'):
        return f

def cohesion_measures(measureType, n_gram):
    if(measureType == 'glue'):
        return glue(n_gram)
    if(measureType == 'dice'):
        return dice(n_gram)
    if(measureType == 'mi'):
        return mi(n_gram)
    if(measureType == 'phi'):
        return phi(n_gram)
    if(measureType == 'logLike'):
        return logLike(n_gram)
    
    return glue(n_gram) # glue is the default measure

def calculateOmegas():

    for i in range(6, 0, -1): # because there isn't any cohesion for zero
        for n_gram in n_grams[i]: # in dictionary entry, and inside that entry cohesions[i][ngram] calculate the cohesion of n_grams
            
            gram_cohesion =  cohesion_measures(COHESION_MEASURE, n_gram) 
            
            cohesions[i][n_gram] = gram_cohesion
            
            if i > 1: # because the 2-gram does have not a cohesion for 1-word                
                omegaMinus1[i][n_gram] = [cohesion_measures(COHESION_MEASURE, n_gram[1:])]
                omegaMinus1[i][n_gram].append(cohesion_measures(COHESION_MEASURE, n_gram[:i]))
            
            try: 
                omegaPlus1[i-1][n_gram[1:]].append(gram_cohesion)
    
            except KeyError:
                omegaPlus1[i-1][n_gram[1:]] = [gram_cohesion]
            
            try: 
                omegaPlus1[i-1][n_gram[:i]].append(gram_cohesion)
            
            except KeyError:
                omegaPlus1[i-1][n_gram[:i]] = [gram_cohesion]


def readCorpus():
    # Find a set of word characters ([\w'’-])     + means that has at least one ocurrence or more of words followed or not by '’-
    # | means or (word characters or punctuation)    
    # where the punctuation is [      ] within this set
    # [ ; : ! ? < > & ( )  \[    \]   to interpret these not as metacharacters but as [  ] characters itself  
    # [ ; : ! ? < > & ( )  \[  \]   \" to not interpret " has a close sign
    # [ ; : ! ? < > & ( )  \[  \]  \"  \. , = / \\ (to not interpret \ as an escaoe signal)]
    # Not adding spaces on ' and - when they are attached to words
    # And also not substituting isolated '’- with white spaces 
    
    regex = re.compile("[\w'’-]+|[;:!?<>&\(\)\[\]\"\.,=/\\\^\$\*\+\|\{\}]|[\S'’-]+")
    
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

    return text_split_list


text_split_list = readCorpus()

words, bigrams, trigrams, fourgrams, fivegrams, sixgrams, sevengrams = find_n_grams(text_split_list)

n_grams = {
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

cohesions = {
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
          
# Main method - Find relevant multi-word units (mwu)

calculateOmegas()
mwu = set([])


for i in range(1, 6):  # starts in 1 till 6 (inclusively)
    for n_gram in n_grams[i]:
        
        n_gram_cohesion = cohesions[i][n_gram]
        
        if n_grams_freq[i][n_gram] >= 2:    
            
            if i == 1:
                if n_gram_cohesion > max(omegaPlus1[i][n_gram]):
                    mwu.add((n_gram_cohesion, " ".join(n_gram)))
            else:
                x = max(omegaMinus1[i][n_gram])
                y = max(omegaPlus1[i][n_gram])
                
                if n_gram_cohesion > (x + y) / 2:
                    mwu.add((n_gram_cohesion, " ".join(n_gram)))
                
                
with open("mwu.txt", "w", encoding="utf-8") as file:
    for relevant_expression in mwu:
        file.write(str(relevant_expression) + "\n")
    
    