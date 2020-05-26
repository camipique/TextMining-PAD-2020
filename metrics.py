# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:15:05 2020

@author: 
    Carlos Quendera 49946
    David Pais 50220
"""
import os, json

CORPUS = ["corpus2mw", "corpus4mw"]
COHESION_MEASURES = ["glue", "dice", "mi", "phi", "log_like"]

# Change manually ( how many of the 200 random selected REs we considered as real REs)
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
    
    print("\n\t\t\t------------------------------ {} ------------------------------\t\t\t\n".format(corpus))

    for cohesion in COHESION_MEASURES:
        print("{}:\n".format(cohesion))
        extracted_re_with_cohesion = read_extractor(corpus, cohesion)
        count = 0
        
        with open('extracted-re-hand.txt') as f:
            line = f.readline()
            
            while line:        
                
                if line.strip() in extracted_re_with_cohesion:
                    count += 1
                    
                line = f.readline()
                
        precision = precisions[corpus][cohesion]
        recall = count/200
        f1_score = 2 * precision * recall / (precision + recall)
    
        print("\t\tPrecision = {}".format(precision))
        print("\t\tRecall = {}".format(recall))
        print("\t\tF1 score = {}\n".format(round(f1_score, 4)))