# -*- coding: utf-8 -*-
"""
PAD Project - Text Mining

@author: Carlos Quendera 49946
@author: David Pais 50220
"""

# Part II a) - Automatic Extraction of Explicit and Implicit Keywords

# Re for regular expression matching operations, os for using operating system dependent functionality, math for mathematic operations, time to measure program's execution time, json to receive json file to input data in json format for keywords extraction phase
# Heapq to see n_grams with the highest tf_idf, statistics for the mean and median function, numpy for data management
import re, os, math, time, json, heapq, statistics, numpy as np

# Nltk to work with human language data (frequency of words and n_grams in a text, ngrams function used for the single keywords)
from nltk import FreqDist
from nltk.util import everygrams, ngrams

# Returns the actual time of the system in seconds
start_time = time.time()

CORPUS_FOLDER_PATH = "corpusTest/" # Change corpus to extract keywords from that corpus
COHESION_MEASURE = "glue" # Same coehesion measure of the RE extracted using the extractor.py to get the keywords using that measure, or load a existent measure file of RE extracted
WEIGTH = "median_syllables" # mean_length, median_length, mean_syllables or median_syllables
THRESHOLD = 0.4

# Read the full corpus to do the inter-document proximity and intra-document proximity 
def read_corpus():

    print("Reading corpus...")
    
    regex = re.compile("[\w'’-]+|[;:!?<>&\(\)\[\]\"\.,=/\\\^\$\*\+\|\{\}\%\'\’\-\“\”\—\–\§\¿?¡!]|[\S'’-]+") 
    
    docs_size = dict() # size of each document
    n_grams_freq_corpus_doc = dict() # n_grams freq inside each document of the corpus
    n_grams_doc = dict() # to which documents a n_gram belongs to
    terms_doc = dict() # uni-grams of each doc
    docs_text = dict() # to use for intra-document frequency
    docs_re = dict()

    # For every file in the input corpus folder
    for file_name, i in zip(sorted(os.listdir(CORPUS_FOLDER_PATH), key = len), range(len(os.listdir(CORPUS_FOLDER_PATH)))):          
    
        with open(CORPUS_FOLDER_PATH + file_name , "r", encoding="utf-8") as file:
            text = file.read()
            
            # remove doc and br identification strings
            text_without_doc = re.sub('<doc(.*?)>|<br(.*?)>' , " ", text)
            
            # find the regex defined in text
            text_list = re.findall(regex, text_without_doc)
            
            terms = list(ngrams(text_list, 1))
            n_grams = list(everygrams(text_list, min_len=2, max_len=6)) # here we do not need the sevengrams because we already have RE calculated from the extractor.py and the n_grams are from n = 2 to n = 6
            
            for doc_n_gram in n_grams: # documents a n_gram belongs to, if that n_gram is being processed for the first time, create a dictionary entry position with an empty set
                n_grams_doc.setdefault(doc_n_gram , set()).add(i)
            
            for term in terms: # document that each uni-gram belongs to
                terms_doc.setdefault(term , set()).add(i)
            
            n_grams_freq_corpus_doc[i] = FreqDist(n_grams)
            
            docs_size[i] = len(text_list)
                
            docs_re[i] = set() # relevant expressions of each document 
            
            docs_text[i] = text_list # text of each document of the corpus
            
    print("Corpus read in %s seconds\n" % (time.time() - start_time))

    return docs_size, n_grams_freq_corpus_doc, n_grams_doc, docs_text, docs_re, i + 1, terms_doc # i + 1 is the number of documents in corpus (since i = 0 in the beginning of the loop)

# Read the json file with the RE extracted using the extractor.py
def read_extractor():
    
    with open(os.path.join("mwu", "{}-{}-mwu.txt".format(CORPUS_FOLDER_PATH[:-1], COHESION_MEASURE)), "r", encoding="utf-8") as file:
        extracted_re = json.load(file)
        
    return extracted_re

# Find to each documents a relevant expression belongs to
def find_docs_re(n_grams_doc, extracted_re, n_grams_freq_corpus_doc, docs_re):
        
    average_prob = dict() # to store inter-document proximity p(A,.) average probability of A in every document of the corpus
    
    for corpus_re in extracted_re: # for each relevant extracted using the extractor.py
    
        n_docs = n_grams_doc[corpus_re] # get the set of docs a relevant expression is in (the n_gram relevant expression is in, set of number of documents for each n_gram relevant expression)
        mean_prob = 0
        average_prob[corpus_re] = 0

        for doc in n_docs: # for every doc in the documents the relevant expression is in
             
            docs_re[doc].add(corpus_re)
            
            mean_prob += n_grams_freq_corpus_doc[doc][corpus_re] / docs_size[doc] # calculate p(A, di)


        mean_prob *= 1 / n_documents # n_documents in the corpus, to calculate p(A,.)

        average_prob[corpus_re] = mean_prob
         
    return docs_re, average_prob

# Count syllables in a relevant expression using number of vowels in words of relevant expression
def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy" # energy  e - ner - gy
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels: # two consecutive vowels do not count as a vowel
            count += 1
    if count == 0:
        count += 1
    return count

# Main method

docs_size, n_grams_freq_corpus_doc, n_grams_doc, docs_text, docs_re, n_documents, terms_doc = read_corpus()

extracted_re_with_cohesion = read_extractor()

# Filter to keep RE which cohesions are bigger than 0.4, if this value is too high, the part of len(docs_re[doc]) > 10 it will not be verified for any of the docs
if COHESION_MEASURE != "mi" or COHESION_MEASURE != 'log_like': # these measures range is ]-oo, +oo[
    # To try with different approaches, substitute with statitics.mean of sylabble_count, or  statistics.mean/statistics.median of len(term) to experiment with the average length of relevant expressions, use the same as the WEIGHT criteria
    extracted_with_threshold = {k: v for k,v in extracted_re_with_cohesion.items() if v * statistics.median([syllable_count(term) for term in k]) > THRESHOLD } 
else:
     # To try with different approaches, substitute with statitics.mean of sylabble_count, or  statistics.mean/statistics.median of len(term) to experiment with the average length of relevant expressions
    extracted_with_threshold = {k: v for k,v in extracted_re_with_cohesion.items() if v * statistics.median([syllable_count(term) for term in k]) > -17 }

extracted_re = list([tuple(re.split(' ')) for re in extracted_with_threshold])     

docs_re, average_prob = find_docs_re(n_grams_doc, extracted_re, n_grams_freq_corpus_doc, docs_re)

print("Calculating explicit keywords...") 

# Calculate Tf-Idf of RE of each document for finding explicit document keywords
 
tf_idf = dict()
tf_idf_terms = dict()
top_tf_idf = dict()
top_tf_idf_terms = dict()
chosen_docs = set() 

for doc in docs_re:
    
    min_re = 10 # choose only documents with more than 10 relevant expressions
    
    if COHESION_MEASURE == "mi" or COHESION_MEASURE == 'log_like':
        min_re = 7
    
    if len(docs_re[doc]) > min_re:
        chosen_docs.add(doc)
        tf_idf[doc] = dict()
        tf_idf_terms[doc] = dict()
        top_tf_idf[doc] = dict()
        top_tf_idf_terms[doc] = dict()
    
    if len(chosen_docs) == 5: # 5 documents were choosen
        break
    
# Returns a universal function (ufunc) object, add broadcasting to a python function, faster than applying a for loop or a map    
get_len = np.frompyfunc(len,1,1)    
get_syllables = np.frompyfunc(syllable_count, 1 , 1)
    
for doc in chosen_docs:
    for relevant_expression in docs_re[doc]:
        if WEIGTH == "mean_length":
            weigth = statistics.mean(get_len(relevant_expression))
        elif WEIGTH == "median_length":
            weigth = statistics.median(get_len(relevant_expression))
        elif WEIGTH == "mean_syllables":
            weigth = statistics.mean(get_syllables(relevant_expression))
        elif WEIGTH == "median_syllables":
            weigth = statistics.median(get_syllables(relevant_expression)) 
        else:
            print("WEIGTH = {} is not defined, choose between [mean | median | syllables]".format(WEIGTH))
            print("Using median as default\n")
            weigth = statistics.median([len(term) for term in relevant_expression])
        
        tf_idf[doc][relevant_expression] = n_grams_freq_corpus_doc[doc][relevant_expression] / docs_size[doc] * math.log(n_documents /  len( n_grams_doc[relevant_expression])) * weigth  
        
    freqs = FreqDist(docs_text[doc])
    
    for term in docs_text[doc]:                                                                                             
        tf_idf_terms[doc][term] = freqs[term] / docs_size[doc] * math.log(n_documents /  len(terms_doc[(term,)])) * len(term) # or syllable_count(term)

    top_tf_idf[doc] = heapq.nlargest(5, tf_idf[doc], key=tf_idf[doc].get) # get relevant expressions with highest tf_idf
    top_tf_idf_terms[doc] = heapq.nlargest(5, tf_idf_terms[doc], key=tf_idf_terms[doc].get) # get single words with highest tf_idf

# Write explicit keywords to file
with open(os.path.join("keywords","explicit","{}-{}-{}.txt".format(CORPUS_FOLDER_PATH[:-1], COHESION_MEASURE, WEIGTH)), "w", encoding="utf-8") as f:
    for doc in chosen_docs:
        f.write("--- Document number {} ---\n\n".format(doc))
        
        f.write("SINGLE words:\n\n")
        
        for term in top_tf_idf_terms[doc]:
            f.write(term+"\n")
            
        f.write("\nMULTI words:\n\n")
        
        for explicit in top_tf_idf[doc]:
            f.write(" ".join(explicit)+"\n")
        
        f.write("\n")
                   
print("Explicit keywords found in %s seconds\n" % (time.time() - start_time))      
 
# Calculate correlation for finding implicit keywords (semantic proximity)

print("Calculating correlations and intra-proximity...") 

corr = dict()

ip = dict()

# for each intra-document expression (need two distinct occurences dictionaries for the case a n_gram relevant expression is a subgram of the other n_gram)
occurences_A = dict()
occurences_B = dict()

for n_doc_A in chosen_docs: # only RE present in the 5 documents
    for a in top_tf_idf[n_doc_A]:
        for n_doc_B in docs_re:
            if n_doc_A == n_doc_B: # it means we are comparing RE in the same document, we only want RE from other documents.
                continue
            
            for b in docs_re[n_doc_B]:
                
                if a == b:
                    continue
                
                # intersecting both sets, if we don't get an empty set, both explicit keyword and re appear in at least a document
                if n_grams_doc[a] & n_grams_doc[b]:   

                    
                    if (n_doc_A in n_grams_doc[a] & n_grams_doc[b]):   # if the re from other document (n_grams_doc[b]) is also in this document (n_doc_A) it cannot be implicit keyword of this document
                        continue  
        
                    
                    docs_both_size = len(n_grams_doc[a] & n_grams_doc[b])
                    
                    ip_a_b = 1 - (1/ docs_both_size)
                    
                    aux_ip_a_b = 0
                      

                    # Intra-document Proximity (IP)
                        
                    for docs_both in n_grams_doc[a] & n_grams_doc[b]:
                           
                        if docs_both_size == 1:   # since ip_a_b = 1 - (1/ docs_both_size) *  np.sum(dist/farthest), if docs_both_size == 1, ip_a_b = 0    
                            break
                        
                        freq_A = n_grams_freq_corpus_doc[docs_both][a]
                        freq_B = n_grams_freq_corpus_doc[docs_both][b]
                        size_doc = docs_size[docs_both]
                        
                        # calculating farthest
                        
                        c1 = freq_A * (size_doc - freq_B)
                        c2 = ( (freq_A - 1) ** 2  + freq_A - 1) / 2
                        c3 = freq_B * (size_doc - freq_A)
                        c4 = ( (freq_B - 1) ** 2  + freq_B - 1) / 2
                        
                        farthest = c1 - c2 + c3 - c4
                        
                        # calculating dist
                        
                        get_doc_words = docs_text[docs_both]
                        
                        # a
                        
                        n_occurences_A_found = 0
                        counter_value_A = 0
                        
                        first_index_occurence_A = 0
                        last_index_occurence_A = 0
                        
                        occurences_A = {}
        
                        # b
                
                        n_occurences_B_found = 0
                        counter_value_B = 0
                        
                        first_index_occurence_B = 0
                        last_index_occurence_B = 0
                        occurences_B = {}
                        
                        dist = 0
                        

                        for n_doc_words in range(0, len(get_doc_words)):
      
                        
                            if(n_occurences_A_found == freq_A and n_occurences_B_found == freq_B):
                                 break   
                        
                            # if first word of n_gram A was found
                            if(counter_value_A == 0 and a[counter_value_A] == get_doc_words[n_doc_words]):
                                first_index_occurence_A = n_doc_words
                                
                            # if first word of n_gram B was found
                            if(counter_value_B == 0 and b[counter_value_B] == get_doc_words[n_doc_words]):
                                first_index_occurence_B = n_doc_words    
                                
                            
                            # a
                            
                            if(a[counter_value_A] == get_doc_words[n_doc_words]): 
                                if(counter_value_A == len(a) - 1 and a[counter_value_A] == get_doc_words[n_doc_words]):
                                    last_index_occurence_A = n_doc_words
                                    n_occurences_A_found += 1
                                    occurences_A.update({n_occurences_A_found : (first_index_occurence_A, last_index_occurence_A)})
                                    counter_value_A = 0
                 
                                else:
                                    counter_value_A += 1
                                   
                            else:
                                counter_value_A = 0                    
      
                            # b
                        
                            if(b[counter_value_B] == get_doc_words[n_doc_words]):
                                if(counter_value_B == len(b) - 1 and b[counter_value_B] == get_doc_words[n_doc_words]):
                                    last_index_occurence_B = n_doc_words
                                    n_occurences_B_found += 1
                                    occurences_B.update({n_occurences_B_found : (first_index_occurence_B, last_index_occurence_B)})
                                    counter_value_B = 0
                                
                                else:   
                                    counter_value_B += 1
                                   
                            else:
                                counter_value_B = 0
                    
                        # This is explained in detail with examples in the report    
                        for n_occurence_A in occurences_A:
                            dist_min = math.inf
                            for n_occurence_B in occurences_B:
                                if(occurences_B[n_occurence_B][1] > occurences_A[n_occurence_A][1]):
                                    aux_dist = abs(occurences_A[n_occurence_A][1]- occurences_B[n_occurence_B][0] - 1)
                                    
                                if(occurences_A[n_occurence_A][1] > occurences_B[n_occurence_B][1]):
                                    aux_dist = abs(occurences_B[n_occurence_B][1]- occurences_A[n_occurence_A][0] - 1) # if dist = 0, means an n_gram immediately follows the other
                                
                                if(aux_dist < 0): # means a n_gram relevant expression is a subgram of the other n_gram relevant expression
                                    aux_dist = 0
                                        
                                if(aux_dist < dist_min):
                                    dist_min = aux_dist
                             
                            dist += dist_min
                 
                             
                        dist *= 2 # because distance of A to B is the same as distance of B to A   
                
                        aux_ip_a_b += dist / farthest  
                    
                    
                    ip_a_b *= aux_ip_a_b
            
                    ip[(b, a)] = ip_a_b  #(b,a) is the same as (a,b) but just to put it in the right shape for the scores calculation
                    
                    
                    # Inter-document proximity (correlation)
                    
                    cov_a_b = 1 / (n_documents - 1)
                    cov_a_a = 1 / (n_documents - 1)
                    cov_b_b = 1 / (n_documents - 1)
                    
                    aux_cov_ab = 0
                    aux_cov_aa = 0
                    aux_cov_bb = 0
                    
                    for n_doc in range(0, n_documents):
                    
                        aux_cov_ab += (n_grams_freq_corpus_doc[n_doc][a]  / docs_size[n_doc] - average_prob[a]) * (n_grams_freq_corpus_doc[n_doc][b]  / docs_size[n_doc] - average_prob[b])
                        aux_cov_aa += (n_grams_freq_corpus_doc[n_doc][a]  / docs_size[n_doc] - average_prob[a]) * (n_grams_freq_corpus_doc[n_doc][a]  / docs_size[n_doc] - average_prob[a])
                        aux_cov_bb += (n_grams_freq_corpus_doc[n_doc][b]  / docs_size[n_doc] - average_prob[b]) * (n_grams_freq_corpus_doc[n_doc][b]  / docs_size[n_doc] - average_prob[b])

                    cov_a_b *= aux_cov_ab
                    cov_a_a *= aux_cov_aa
                    cov_b_b *= aux_cov_bb

                    corr_a_b = cov_a_b / (math.sqrt(cov_a_a) * math.sqrt(cov_b_b))
                      
                    corr[(b, a)] = corr_a_b #(b,a) is the same as (a,b) but just to put it in the right shape for the scores calculation

print("Calculated correlations and intra-proximity in %s seconds\n" % (time.time() - start_time)) 


print("Calculating implicit keywords...") 

# Scores
scores = dict()
top_scores = dict()

for doc in chosen_docs:
    # corr and ip values of an explicit keyword and of a re that are in the same doc is 0
    scores[doc] = dict()
    for relevant_expression in extracted_re:
        score = 0
        
        if " ".join(relevant_expression) in " ".join(docs_text[doc]): # if we have equal top scores, ex: 5th top score is from a expression outside and one inside that doc (it can be from one inside doc if all scores is 0 as a case we tested)
            continue
        
        for i, explicit_keyword in enumerate(top_tf_idf[doc]):        
            
            if corr.get((relevant_expression, explicit_keyword)) and ip.get((relevant_expression, explicit_keyword)):
                score += (corr[(relevant_expression, explicit_keyword)] * math.sqrt(ip[(relevant_expression, explicit_keyword)])) / (i+1)
            
        scores[doc][relevant_expression] = score
        
    top_scores[doc] = heapq.nlargest(5, scores[doc], key=scores[doc].get)
    
# Write implicit keywords to a file    
with open(os.path.join("keywords","implicit","{}-{}-{}.txt".format(CORPUS_FOLDER_PATH[:-1], COHESION_MEASURE, WEIGTH)), "w", encoding="utf-8") as f:
    for doc in chosen_docs:
        f.write("--- Document number {} ---\n\n".format(doc))
        for implicit in top_scores[doc]:
            f.write(" ".join(implicit)+"\n")
            
        f.write("\n")
            
print("Calculated implicit keywords in %s seconds\n" % (time.time() - start_time)) 
