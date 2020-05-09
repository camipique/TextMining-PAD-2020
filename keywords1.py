# -*- coding: utf-8 -*-
"""
PAD Project - Text Mining
@author: Carlos Quendera 49946
@author: David Pais 50220
"""

# Part II a) - Automatic Extraction of Explicit and Implicit Keywords

import re, os, math
import numpy as np
from nltk import FreqDist
from nltk.util import everygrams
import time

start_time = time.time()

CORPUS_FOLDER_PATH = "corpus2mwTest2nd/"
COHESION_MEASURE = "glue" # change here to use a different cohesion measure


def n_gram_prob(n_gram, document_words): # [dictionary entry][element inside the dictionary entry]
    return n_grams_freq_doc[i][n_gram] / document_words

def glue(n_gram, document_words):
    f = cohesion(n_gram, 'glue', document_words)
    return n_gram_prob(n_gram, document_words) ** 2 / f # formula of glue (scp) of a n-gram with n > 2 (it can also apply to n = 2)
  
def dice(n_gram, document_words):
    f = cohesion(n_gram, 'dice', document_words)
    return (document_words * n_gram_prob(n_gram, document_words) * 2) / f # formula of dice of a n-gram with n > 2 (it can also apply to n = 2)
  
def mi(n_gram, document_words):
    f = cohesion(n_gram, 'mi', document_words) # n_gram_prob(n_gram) / f will never be 0 since prob and f > 0
    return math.log(n_gram_prob(n_gram, document_words) / f) # formula of mi of a n-gram with n > 2 (it can also apply to n = 2)

def phi(n_gram, document_words):
    avq, avd = cohesion(n_gram, 'phi', document_words) # word count(number of words in corpus) * frequency
    return (( (document_words ** 2 * n_gram_prob(n_gram, document_words)) - avq) ** 2) / avd # formula of phi of a n-gram with n > 2 (it can also apply to n = 2)

def log_l(p, k, m): # auxiliar function for logLike cohesion measure
        return k * math.log(p) + (m - k) * math.log(1-p)

def logLike(n_gram, document_words):
    left_subgram, right_subgram = cohesion(n_gram, 'logLike')
    kf1 = document_words * n_gram_prob(n_gram, document_words)
    kf2 = left_subgram - kf1
    nf1 = right_subgram
    nf2 = document_words - nf1
    pf1 = kf1/nf1 
    pf2 = kf2/nf2
    pf = left_subgram/document_words

    # formula of logLike of a n-gram with n > 2 (it can also apply to n = 2)
    if( (pf1 > 0 and pf1 < 1) and  (pf2 > 0 and pf2 < 1) and (pf > 0 and pf < 1)): # < 1 because of math.log(1-p) of log_l
       return 2 * ( log_l (pf1, kf1, nf1) + log_l (pf2, kf2, nf2) - log_l (pf, kf1, nf1) - log_l(pf, kf2, nf2))
   
    else:
        return -math.inf # to avoid being bigger than other omega plus 1 values that might be smaller, ln domain goes from [-oo, oo]

def cohesion (n_gram, measure, document_words):

    f = 0 # frequency
        
    avq = 0 
    avd = 0
    
    avx = 0
    avy = 0
    
    n_gram_size = len(n_gram)
    
    for i in range(1,  n_gram_size ): # starting in 1 because :1 goes till the start index, so starts in 0. i: starts in the index (1) till the index
        left_subgram = n_gram_prob(n_gram[:i], document_words)
        right_subgram = n_gram_prob(n_gram[i:], document_words)
        
        if(not (measure == 'dice')):
            
            if(measure == 'phi'):
                left_subgram = left_subgram * document_words
                right_subgram = right_subgram * document_words
                avq += left_subgram * right_subgram 
                avd += (left_subgram * right_subgram) * (document_words - left_subgram) * (document_words - right_subgram)        
            elif(measure == 'logLike'):
                avx += document_words * left_subgram
                avy += document_words * right_subgram            
            else:                    
                f += left_subgram * right_subgram
    
        else:
            f += document_words * ( left_subgram + right_subgram)
 
    if(measure == 'phi'):
        avq = avq / (  n_gram_size  - 1 )
        avd = avd / (  n_gram_size  - 1 )
        
        return avq, avd
    
    if(measure == 'logLike'):
        avx = avx / (  n_gram_size  - 1 )
        avy = avy / (  n_gram_size  - 1 )
        
        return avx,avy

    if(measure == 'glue' or measure == 'dice' or measure == 'mi'):
        f = f / (  n_gram_size  - 1 ) # formula of F 
        
        return f

def cohesion_measures(measure_type, n_gram, document_words):    
    if (len(n_gram) == 1):
        return n_gram_prob(n_gram, document_words)
    elif(measure_type == 'glue'):
        return glue(n_gram, document_words)
    elif(measure_type == 'dice'):
        return dice(n_gram, document_words)
    elif(measure_type == 'mi'):
        return mi(n_gram, document_words)
    elif(measure_type == 'phi'):
        return phi(n_gram, document_words)
    elif(measure_type == 'logLike'):
        return logLike(n_gram, document_words)
    else:
        return glue(n_gram, document_words) # glue is the default measure

def read_corpus():
    # Find a set of word characters ([\w'’-])     + means that has at least one ocurrence or more of words followed or not by '’-
    # | means or (word characters or punctuation)    
    # where the punctuation is [      ] within this set
    # [ ; : ! ? < > & ( )  \[    \]   to interpret these not as metacharacters but as [  ] characters itself  
    # [ ; : ! ? < > & ( )  \[  \]   \" to not interpret " has a close sign
    # [ ; : ! ? < > & ( )  \[  \]  \"  \. , = / \\ (to not interpret \ as an escaoe signal)]
    # Not adding spaces on ' and - when they are attached to words
    # And also not substituting isolated '’- with white spaces 
    print("Reading corpus...\n")
    
    regex = re.compile("[\w'’-]+|[;:!?<>&\(\)\[\]\"\.,=/\\\^\$\*\+\|\{\}]|[\S'’-]+")
    
    document = dict()
    n_grams_doc = dict()
    n_grams_freq_doc = dict()
    seq = dict()
    mwu = dict()

    # order by number and not by the order of the underlying operating system
    for file_name, i in zip(sorted(os.listdir(CORPUS_FOLDER_PATH), key = len), range(len(os.listdir(CORPUS_FOLDER_PATH)))):  
        #print(i)
        #print(CORPUS_FOLDER_PATH + file_name)
        with open(CORPUS_FOLDER_PATH + file_name, "r", encoding="utf8") as f:
            text = f.read()
            
            # remove doc identification strings
            text_without_doc = re.sub('<doc(.*?)>' , " ", text)
            
            # find the regex defined in text             
            document[i] =  re.findall(regex, text_without_doc)
            
            n_grams_doc[i] = list(everygrams(document[i], min_len=1, max_len=7)) # invert to iterate from 7-grams to 1-grams

            n_grams_freq_doc[i] = FreqDist(n_grams_doc[i])

            n_grams_doc[i] = sorted(set(n_grams_doc[i]), key = len, reverse = True)
            
            seq[i] = dict()
            
            mwu[i] = set()

    return document, n_grams_doc, n_grams_freq_doc, seq, mwu, i # n documents - 1

documents, n_grams_doc, n_grams_freq_doc, seq, mwu, i = read_corpus()

print("Corpus read in %s seconds\n" % (time.time() - start_time))

get_size = np.frompyfunc(len,1,1)

with open("mwu2ndpart.txt", "w+", encoding="utf-8") as file: # w+ for both reading and writting file, overwritting the file
    for i in range(0, i + 1 ): # calculate RE for all documents and select 5 documents   
        one_gram_index = np.argmax(get_size(n_grams_doc[i]) < 2) # for n-gram with n > 2, because the cohesion is not calculated for n = 1
        for n_gram_index in range(0, one_gram_index):          
            
            document_words = len(n_grams_doc[i])
            
            get_entry = seq[i].get
            
            cohesion_gram = cohesion_measures
            
            add = mwu[i].add
            
            n_gram = n_grams_doc[i][n_gram_index]

            left_gram = n_gram[:len(n_gram) - 1]

            right_gram = n_gram[1:]

            n_gram_freq = n_grams_freq_doc[i][n_gram]
            
            n_gram_cohesion = cohesion_gram(COHESION_MEASURE, n_gram, document_words)
            
            # since we start from both, we only assign values to n-1 levels, since we don't need the values of the cohesion of sevengrams stored
            left_gram_freq = n_grams_freq_doc[i][left_gram]
            left_gram_cohesion =  cohesion_gram(COHESION_MEASURE, left_gram, document_words) # E.g. United States of America - United States Of
            
            right_gram_freq = n_grams_freq_doc[i][right_gram]
            right_gram_cohesion = cohesion_gram(COHESION_MEASURE, right_gram, document_words) # E.g. United States of America - States of America
        
        
            # left sub_gram        
            if get_entry(left_gram):         
                max_cohesion = seq[i][left_gram][2]    
                if (n_gram_cohesion >  max_cohesion):
                    seq[i][left_gram][2] = n_gram_cohesion 

            else:
                seq[i][left_gram] = [left_gram_freq, left_gram_cohesion, n_gram_cohesion]
            
            # right sub_gram
            if  get_entry(right_gram):
                    max_cohesion = seq[i][right_gram][2]
                    if(n_gram_cohesion > max_cohesion):
                        seq[i][right_gram][2] = n_gram_cohesion    
            else:
                seq[i][right_gram] = [right_gram_freq, right_gram_cohesion, n_gram_cohesion]
      
            # Find Relevant Expressions
            if(len(n_gram) < 7):
                if(n_gram_freq >= 2): # If the n_gram appears at least 2 times in corpus
    
                    if(len(n_gram) == 2):
                        if(n_gram_cohesion > seq[i][n_gram][2]):  
                            add((n_gram))   #" ".join(n_gram)))
        
                    else:           
                        x = max(left_gram_cohesion, right_gram_cohesion)
                        y = seq[i][n_gram][2]
         
                        if  (n_gram_cohesion > (x + y) / 2 ):
                            add((n_gram))  #" ".join(n_gram)))

            
        #file.write(str(mwu[i]) + '\n') # maybe we don't need this?

print("--- Program ended in %s seconds ---" % (time.time() - start_time))

# Calculate correlation for finding implicit keywords (semantic proximity)

average_doc_freq_A = 0
average_doc_freq_B = 0
cov = 0
corr = set()

for f in range(0, 5): # only RE present in the 5 documents 
    for g in range(0, len(mwu[f])):
        for h in range(0, len(mwu)):
            for j in range(0, len(mwu[h])):
                if h == f: # it means we are comparing RE in the same document, we only want RE from other documents.
                    continue
                    
                document_re = list(mwu[f])
                document_outside_re = list(mwu[h]) # Look for all RE outside the document
                re_document = document_re[g]
                re_outside = tuple(document_outside_re[j])
                
                # if A (always inside the document) and B (outside) never appear together in at least one document, the correlation is 0.
                
                for l in range(0, i + 1): # if both RE appear both in a document
                    if(n_grams_freq_doc[l][re_document] > 0 and n_grams_freq_doc[l][re_outside] > 0):
                        for m in range(0, i + 1):
                            for n in range(0, i + 1):
                                average_doc_freq_A += n_grams_freq_doc[n][re_document] / len(n_grams_doc[n])
                                average_doc_freq_B += n_grams_freq_doc[n][re_outside] /  len(n_grams_doc[n])
                            

                            cov = np.sum( (n_grams_freq_doc[m][re_document] - average_doc_freqA/(i + 1) ) * (n_grams_freq_doc[m][re_outside] - average_doc_freqB/(i + 1)))
                        
                        cov = cov/ i # i + 1 - 1 = i
                        corr.add((re_document, re_outside, cov))
                    break  # we already know both expressions appear at least once in the document
                
                average_doc_freqA = 0
                average_doc_freqB = 0
                cov = 0
                
    


                
# Calculate Tf-Idf of RE of each document for finding explicit document keywords
       
# After this calculate the score to determine the implit keywords of each document         
            
