import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import scraper
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

#import and apply bert to classify questions + answers from new_dataset.csv
entries = open("new_dataset.csv", "r")
    


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
for i in range():
    #encode every entry
    #text = ''

# Tokenize and encode the text
encoding = tokenizer.encode(text)
print("Token IDs:", encoding)

# Convert token IDs back to tokens
tokens = tokenizer.convert_ids_to_tokens(encoding)
print("Tokens:", tokens)
