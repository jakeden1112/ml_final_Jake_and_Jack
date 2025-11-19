import pandas as pd
import re
import csv
'''
For our first step, we will be using a preexisting dataset from hf, which we will simplify and make the model able to work with it
Later on, we plan to create our own custom dataset that will have more questions (hf one only has half of questions asked)
'''
def hf_dataframe():
    df = pd.read_json("hf://datasets/openaccess-ai-collective/jeopardy/data/train.jsonl", lines=True)
    return df
df = hf_dataframe()

print(df['answer'].count())
print(df['answer'].nunique())

#print(df.head(20))

qa = df[["question","answer"]]

#print(qa.head(20))

test = qa.loc[0,"question"]

print(test)

#adapted from google's ai response
def cleaner(text):
    clean = re.sub(r'[^\w\s]', '', str(text))
    clean = clean.lower()
        
    return clean


print(cleaner(test))

qa_clean = qa.map(cleaner)

def occ_dict_maker (df, column):
    #we want to pull out each word and how many times it shows up 
    #we will opt to remove common words that add minimal information (the, for, is, etc.)
    occDict = {}
    temp = ''
    for col in range(100):#df[column].count()):
        for ch in str(df.loc[col,"question"]):
            if ch == ' ':
                if not("hrefhttpwwwjarchivecommedia" in temp):
                    temp = temp.replace(' ', '')
                    if temp in occDict:
                        occDict[temp]+=1
                    else:
                        occDict[temp]=1
                temp = ''
                
            temp = temp+ch
            
        if not("hrefhttpwwwjarchivecommedia" in temp):
            temp = temp.replace(' ', '')
            if temp in occDict:
                occDict[temp]+=1
            else:
                occDict[temp]=1
        temp = ''

    return occDict

occ_dict = occ_dict_maker(qa_clean,'question')
print(occ_dict)

#save the dictionary to a csv so we don't have to make it everytime
def dict_to_csv (occ_dict):
    with open('word_occurences.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['word','occurences'])
        for key in occ_dict:
            
            writer.writerow([key, occ_dict[key]])


dict_to_csv(occ_dict)





