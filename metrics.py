# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:15:05 2020

@author: 
    Carlos Quendera 49946
    David Pais 50220
"""
import os, json

CORPUS = ["corpus4mw"]
COHESION_MEASURES = ["glue", "dice", "mi", "phi", "log_like"]

# Change manually
precisions = {
    'corpus2mw': {
        'glue': 103/200,
        'dice': 93/200,
        'mi': 81/200,
        'phi': 97/200,
        'log_like': 71/200         
        },
    'corpus4mw': {
        'glue': 89/200,
        'dice': 64/200,
        'mi': 62/200,
        'phi': 85/200,
        'log_like': 50/200
        }
    }

def read_extractor(corpus, cohesion):
    
    with open(os.path.join("tested-mwus", "{}-{}-mwu.txt".format(corpus, cohesion)), "r", encoding="utf-8") as file:
        extracted_re = json.load(file)
        
    return extracted_re

for corpus in CORPUS:

    for cohesion in COHESION_MEASURES:
        
        extracted_re_with_cohesion = read_extractor(corpus, cohesion)
        count = 0
        
        with open('{}-re-hand'.format(corpus)) as f:
            line = f.readline()
            
            while line:        
                
                if line.strip() in extracted_re_with_cohesion:
                    count += 1
                    
                line = f.readline()
                
        precision = precisions[corpus][cohesion]
        recall = count/200
        f1_score = 2 * precision * recall / (precision + recall)
    
        print("Precision for {} with {} as cohesion measure is {}".format(corpus, cohesion, precision))
        print("Recall for {} with {} as cohesion measure is {}".format(corpus, cohesion, recall))
        print("F1 score for {} with {} as cohesion measure is {}\n".format(corpus, cohesion, round(f1_score, 4)))