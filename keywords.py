# -*- coding: utf-8 -*-
"""
PAD Project - Text Mining

@author: Carlos Quendera 49946
@author: David Pais 50220
"""

# Part II a) - Automatic Extraction of Explicit and Implicit Keywords

import re, os, time, json, heapq, statistics, math
from nltk import FreqDist
from nltk.util import everygrams

start_time = time.time()

CORPUS_FOLDER_PATH = "test/"  # and that we need to change the measure on the extractor file and here to load the file we extracted of that measure
COHESION_MEASURE = "glue" # just here to don't forget to talk in the report about running the other file with the measure we want before running keywords

def read_corpus():

    print("Reading corpus...\n")
    
    regex = re.compile("[\w'’-]+|[;:!?<>&\(\)\[\]\"\.,=/\\\^\$\*\+\|\{\}\%\'\’\-\...\“\”\—\–\§\¿?¡!]|[\S'’-]+") 
    
    docs_size = dict()
    n_grams_freq_corpus_doc = dict()
    n_grams_doc = dict() # to which document a n_gram belongs
    docs_text = dict() # to use for intra-document frequency
    docs_re = dict()

    
    for file_name, i in zip(sorted(os.listdir(CORPUS_FOLDER_PATH), key = len), range(len(os.listdir(CORPUS_FOLDER_PATH)))):          
            # corpus2mw     
        with open(CORPUS_FOLDER_PATH + file_name , "r", encoding="utf-8") as file:
            text = file.read()
            
            # remove doc identification strings
            text_without_doc = re.sub('<doc(.*?)>|<br(.*?)>' , " ", text)
            
            # find the regex defined in text
            text_list = re.findall(regex, text_without_doc)
             
            n_grams = list(everygrams(text_list, min_len=2, max_len=6))
            
            for doc_n_gram in n_grams:
            
                n_grams_doc.setdefault(doc_n_gram , set()).add(i)
            
            #if n_grams_doc.get(doc_n_gram     )
            #n_grams_doc.update({doc_n_gram : i for doc_n_gram in n_grams})
            
            n_grams_freq_corpus_doc[i] = FreqDist(n_grams)
            
            docs_size[i] = len(text_list)
                
            docs_re[i] = dict()
            
            docs_text[i] = text_list


    return docs_size, n_grams_freq_corpus_doc, n_grams_doc, docs_text, docs_re, i + 1


def read_extractor():
    
    with open(os.path.join("mwu", "{}-{}-mwu.txt".format(CORPUS_FOLDER_PATH[:-1], COHESION_MEASURE)), "r", encoding="utf-8") as file:
        extracted_re = json.load(file)
        
    return extracted_re

def find_docs_re(n_grams_doc, extracted_re, n_grams_freq_corpus_doc, docs_re):
        
    for corpus_re in extracted_re:
    
        n_docs = n_grams_doc[corpus_re]

        for doc in n_docs:
            
      
            docs_re[doc][corpus_re] =  n_grams_freq_corpus_doc[doc][corpus_re]

            #else:
            #    docs_re[doc][corpus_re] = {corpus_re : n_grams_freq_corpus_doc[doc][corpus_re]}
                
        
        #ocs_re[re_doc].update({corpus_re : n_grams_freq_corpus_doc[re_doc][corpus_re]})
        
    return docs_re
    

def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy" # energy  e - ner - gy
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if count == 0:
        count += 1
    return count



# Main method

docs_size, n_grams_freq_corpus_doc, n_grams_doc, docs_text, docs_re, n_documents = read_corpus()



extracted_re = read_extractor()

extracted_re = list([tuple(re.split(' ')) for re in extracted_re])     

docs_re = find_docs_re(n_grams_doc, extracted_re, n_grams_freq_corpus_doc, docs_re)

print("Corpus read in %s seconds\n" % (time.time() - start_time))





                
# Calculate Tf-Idf of RE of each document for finding explicit document keywords

           
tf_idf = dict()
chosen_docs = set() # we choose random documents with more than 10 relevant expressions in the corpus 



for doc in docs_re:

    if len(docs_re[doc]) > 10:
        chosen_docs.add(doc)
        tf_idf[doc] = dict()
    
    if len(chosen_docs) == 5:
        break
    
    


#for i in range(0, n_documents):

#for i in range (0, 5):

for doc in chosen_docs:
    
    for relevant_expression in docs_re[doc]:
    
         # n_words = [len(term) for term in relevant_expression]
         # n_syllables = [syllable_count(term) for term in relevant_expression]
        
        n_syllables = syllable_count(" ".join(relevant_expression))                   
                                                                                                                                                                       
        #statistics.median(n_syllables), statistics.mean(n_words), and others
        tf_idf[doc][relevant_expression] = (docs_re[doc][relevant_expression]/docs_size[doc]) * math.log(n_documents /  len( n_grams_doc[relevant_expression])) * n_syllables  

               
        
        
        
        
"""   
    if prob_doc_re.get(relevant_expression):
             prob_doc_re[relevant_expression][re_doc] = docs_re[re_doc][relevant_expression]/docs_size[re_doc]
    else:
             prob_doc_re[relevant_expression] = {re_doc:  n_grams_freq_corpus_doc[re_doc][relevant_expression]/docs_size[re_doc]}
             
"""


"""
for relevant_expression in aux:
    for x in aux[relevant_expression]:
        
        syllables = []
        
        for i in relevant_expression:
            syllables.append(syllable_count(i))
        
        tf_idf_value = aux[relevant_expression][x] * math.log(n_documents / len(aux[relevant_expression])) * syllable_count(" ".join(relevant_expression))
        
        if tf_idf.get(x):
            tf_idf[x][relevant_expression] = tf_idf_value
        else:
            tf_idf[x] = { relevant_expression: tf_idf_value }

# choose the highest 5 for each doc
doc_top_explicit = {}     
   
for doc in tf_idf:
    doc_top_explicit[doc] = heapq.nlargest(5, tf_idf[doc], key=tf_idf[doc].get)
    

print("Program ended in %s seconds" % (time.time() - start_time))    

"""