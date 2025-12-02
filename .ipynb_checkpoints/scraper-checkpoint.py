import pandas as pd
import re
import csv
import requests 
import time


'''
For our first step, we will be using a preexisting dataset from hf, which we will simplify and make the model able to work with it
Later on, we plan to create our own custom dataset that will have more questions (hf one only has half of questions asked)
'''
def hf_dataframe():
    df = pd.read_json("hf://datasets/openaccess-ai-collective/jeopardy/data/train.jsonl", lines=True)
    return df



def cleaner(text):
    #adapted from google's ai response
    clean = re.sub(r'[^\w\s]', '', str(text))
    clean = clean.lower()
    clean = clean.replace('  ', ' ')
    clean = clean.replace('  ', ' ')
    clean = clean.replace('  ', ' ')
    return clean





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

def answer_dict_maker(df, column):
    answerDict = {}
    for col in range(df[column].count()):
        temp = df.loc[col,column]
        if temp in answerDict:
            answerDict[temp]+=1
        else:
            answerDict[temp]=1
    return answerDict

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
            #we can just worry about words of length > 2
            if int(row[1]) > low_bound and int(row[1]) < up_bound and len(str(row[0])) > 2 :
                basket_list.append(row[0])
                output_dict[row[0]] = int(row[1])

        return basket_list, output_dict

def find_good_questions(df, ans_list, fname):
    q_list = []
    with open(fname, 'w', newline = '', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["question number"])
        for i in range(0,len(df)):
            if df.loc[i,"answer"] in ans_list:
                writer.writerow([i])
                q_list.append(i)
                print (i)

    

    return q_list

def retrieve_good_qs(fname):
    good_qs = []
    with open(fname, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            good_qs.append(row[0])

    return good_qs

def get_page_contents(page_name):
    page = requests.get(page_name)
    return page.text

def remove_html(text):
    while "<" in text or ">" in text:
        pos1 = text.find("<")
        pos2 = text.find(">")
        text = text[:pos1] + text[pos2+1:]

    while "&amp;" in text:
        text = text.replace('&amp;','&')
    return text
    

def page_scraper(page_name):
    page_text = get_page_contents(page_name)
    with open('new_dataset.csv', 'a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
    
        category_list = [] #always 6 long (for single and double, final is just 1 category)

        #gets show # and airdate from the title of the html page
        pos1 = page_text.find("<title>")
        pos2 = page_text.find("</title>")
        title = page_text[pos1:pos2]
        pos1 = title.find("Show #")+6
        show_num = title[pos1:pos1+4]
        pos2 = title.find("aired ")+6
        airdate = title[pos2:pos2+10]

        #gets the list of categories for the first round
        pos1 = page_text.find("jeopardy_round")
        page_text = page_text[pos1:]
        
        offset = 0 #we will save all 12 categories to the list; first questions of double jeopardy will then be 6th on list
        round_name = "Jeopardy"
        for round in range(2):
            
            for i in range(6):
                pos1 = page_text.find("category_name")+len("category_name")+2
                page_text = page_text[pos1:]
                pos2 = page_text.find("</td>")
                category_list.append(remove_html(page_text[:pos2]))

            
            for i in range(30):
                daily_double = 0
                pos1 = page_text.find("clue_value")+len("clue_value")+3
                if(pos1 > page_text.find("clue_value_daily_double") and page_text.find("clue_value_daily_double") > 0):
                    pos1 = page_text.find("clue_value_daily_double")+len("clue_value_daily_double")+7
                    daily_double = 1
                        
                page_text = page_text[pos1:]
                pos2 = page_text.find("</td>")
                
                if daily_double == 1:
                    if i % 6 == 0:
                        if round_name == "Jeopardy":
                            value = int(value)+200
                        else:
                            value = int(value)+400
                else:
                    value = page_text[:pos2]
                
                pos1 = page_text.find("clue_text")+len("clue_text")+2
                page_text = page_text[pos1:]
                pos2 = page_text.find("</td>")
                question = page_text[:pos2]
                question = remove_html(question)
    
                pos1 = page_text.find("correct_response")+len("correct_response")+2
                page_text = page_text[pos1:]
                pos2 = page_text.find("</em>")
                answer = page_text[:pos2]
                answer = remove_html(answer)
                
                writer.writerow([show_num, airdate, round_name, category_list[(i%6)+offset], value , question, answer])
            if round == 1:
                break
            
            round_name = "Double Jeopardy"
            pos1 = page_text.find("double_jeopardy_round")
            page_text = page_text[pos1:]
            offset = 6

        round_name = "Final Jeopardy"
        pos1 = page_text.find("final_jeopardy_round")
        page_text = page_text[pos1:]

        

        pos1 = page_text.find("category_name")+len("category_name")+2
        page_text = page_text[pos1:]
        pos2 = page_text.find("</td>")
        category_list.append(remove_html(page_text[:pos2]))
        
        pos1 = page_text.find("clue_text")+len("clue_text")+2
        page_text = page_text[pos1:]
        pos2 = page_text.find("</td>")
        question = page_text[:pos2]
        question = remove_html(question)

        pos1 = page_text.find("correct_response")+len("correct_response")+2
        page_text = page_text[pos1:]
        pos2 = page_text.find("</em>")
        answer = page_text[:pos2]
        answer = remove_html(answer)

        value = "final" #maybe make higher number when turning to df, like 3000 or 4000
        
            
        writer.writerow([show_num, airdate, round_name, category_list[12], value , question, answer])

#uses other function to scrape multiple pages
def site_scraper(start, end):
    for page_num in range(start, end):
        print(page_num)
        page_scraper("https://j-archive.com/showgame.php?game_id="+str(page_num))
        time.sleep(1)

if __name__ == '__main__':

    with open('new_dataset.csv', 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["show #", "airdate","round","category","value", "question","answer"])
        
    
    #page_scraper("https://j-archive.com/showgame.php?game_id=9322")
    #site_scraper(9317,9322)
    site_scraper(100,110)
    
    '''
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
    

    
    
    
    #occ_dict = occ_dict_maker(qa_clean,'question')
    #print(occ_dict)
    
    #dict_to_csv(occ_dict, 'word_occurences.csv')
    
    basket, test_dict  = create_basket('word_occurences.csv',30,500)
    
    #print(test_dict)
    print(len(test_dict))
    print(len(basket))
    '''
    
    
    print("done")





