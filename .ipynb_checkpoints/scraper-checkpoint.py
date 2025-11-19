import pandas as pd
import re
'''
For our first step, we will be using a preexisting dataset from hf, which we will simplify and make the model able to work with it
Later on, we plan to create our own custom dataset that will have more questions (hf one only has half of questions asked)
'''

df = pd.read_json("hf://datasets/openaccess-ai-collective/jeopardy/data/train.jsonl", lines=True)

print(df['answer'].count())
print(df['answer'].nunique())

#print(df.head(20))

qa = df[["question","answer"]]

#print(qa.head(20))

test = qa.loc[0,"question"]

print(test)

#adapted from google's ai response
def cleaner(str):
    clean = re.sub(r'[^\w\s]', '', str)
    return clean


print(cleaner(test))

#we want to pull out each word and how many times it shows up 
#we will opt to remove common words that add minimal information (the, for, is, etc.)
occDict = {}
temp = ''
for col in range(20):#qa['answer'].count()):
    for ch in str(qa.loc[col,"question"]):
        if ch == ' ':
            if temp in occDict:
                occDict[temp]+=1
            else:
                occDict[temp]=1
            temp = ''
            
        temp = temp+ch
        
    if temp in occDict:
        occDict[temp]+=1
    else:
        occDict[temp]=1
    temp = ''

print(occDict)





