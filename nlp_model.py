import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import scraper
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

'''
this is old. I don't believe it was used for the final project
'''

#https://www.geeksforgeeks.org/nlp/explanation-of-bert-model-nlp/

#import and apply bert to classify questions + answers from new_dataset.csv
entries = pd.read_csv("new_dataset.csv")
encoded_questions = []
tokens_questions = []

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

for question in entries['question']:
    #encode every entry
    #text = ''
    text = str(question)
    
    #encode text
    encoding = tokenizer.encode(text, add_special_tokens=True)
    
    #add to encoded list
    encoded_questions.append(encoding)
    
    tokens = tokenizer.convert_ids_to_tokens(encoding)
    tokens_questions.append(tokens)

print(encoded_questions[1])
print(tokens_questions[1])


