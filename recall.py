# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:15:05 2020

@author: 
    Carlos Quendera 49946
    David Pais 50220
"""
import os, json

CORPUS_FOLDER_PATH = "corpus4mw/"
COHESION_MEASURES = ["glue", "dice", "mi", "phi", "log_like"]

def read_extractor(cohesion):
    
    with open(os.path.join("tested-mwus", "{}-{}-mwu.txt".format(CORPUS_FOLDER_PATH[:-1], cohesion)), "r", encoding="utf-8") as file:
        extracted_re = json.load(file)
        
    return extracted_re

for cohesion in COHESION_MEASURES:
    
    extracted_re_with_cohesion = read_extractor(cohesion)
    count = 0
    
    with open('4mw-re-hand') as f:
        line = f.readline()
        
        while line:        
            
            if line.strip() in extracted_re_with_cohesion:
                count += 1
                
            line = f.readline()
    
    print("Recall for {} with {} as cohesion measure is {}/200\n".format(CORPUS_FOLDER_PATH[:-1], cohesion, count))