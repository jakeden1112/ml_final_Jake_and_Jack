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


#adapted from google's ai response
def cleaner(text):
    clean = re.sub(r'[^\w\s]', '', str(text))
    clean = clean.lower()
    clean = clean.replace('  ', ' ')
    clean = clean.replace('  ', ' ')
    clean = clean.replace('  ', ' ')
    return clean


def answer_dict_maker(df, column):
    answerDict = {}
    for col in range(df[column].count()):
        temp = df.loc[col,column]
        if temp in answerDict:
            answerDict[temp]+=1
        else:
            answerDict[temp]=1
    return answerDict


def occ_dict_maker (df, column):
    #we want to pull out each word and how many times it shows up 
    #we will opt to remove common words that add minimal information (the, for, is, etc.)
    occDict = {}
    temp = ''
    for col in range(df[column].count()):
        for ch in str(df.loc[col,column]):
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



#save the dictionary to a csv so we don't have to make it everytime
def dict_to_csv (occ_dict, fname):
    with open(fname, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['word','occurences'])
        for key in occ_dict:
            
            writer.writerow([key, occ_dict[key]])

def create_basket (fname, low_bound, up_bound):
    output_dict = {}
    basket_list = []
    with open(fname, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        
        for row in reader:
            #we don't want to worry about words that are very uncommon and also words that are so common they are in almost every question
            if int(row[1]) > low_bound and int(row[1]) < up_bound :
                basket_list.append(row[0])
                output_dict[row[0]] = int(row[1])

        return basket_list, output_dict

if __name__ == '__main__':

    df = hf_dataframe()

    print(df['answer'].count())
    print(df['answer'].nunique())
    
    #print(df.head(20))
    
    qa = df[["question","answer"]]
    
    #print(qa.head(20))
    
    test = qa.loc[0,"question"]
    
    print(test)

    print(cleaner(test))

    qa_clean = qa.map(cleaner)

    #print(qa_clean.head(20))

    #answer_dict = answer_dict_maker(qa_clean, 'answer')
    
    #dict_to_csv(answer_dict, 'answer_occurences.csv')

    basket, test_dict  = create_basket('Answer_occurences.csv',5,20000)

    print(test_dict)
    print(len(test_dict))
    print(len(basket))
    
    '''
    #occ_dict = occ_dict_maker(qa_clean,'question')
    #print(occ_dict)
    
    #dict_to_csv(occ_dict, 'word_occurences.csv')
    
    basket, test_dict  = create_basket('word_occurences.csv',30,500)
    
    #print(test_dict)
    print(len(test_dict))
    print(len(basket))
    '''
    
    
    print("done")





