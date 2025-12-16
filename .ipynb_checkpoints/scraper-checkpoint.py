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

def our_dataframe():
    df = pd.read_csv("final_dataset_clean_standardized.csv",low_memory=False)
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
    print(page)
    if(page.status_code == 200):
        return page.text
    else:
        return -1

def remove_html(text):
    while "<" in text or ">" in text:
        pos1 = text.find("<")
        pos2 = text.find(">")
        text = text[:pos1] + text[pos2+1:]
        #print(text)

    while "&amp;" in text:
        text = text.replace('&amp;','&')
    return text
    
def remove_spaces(text):
    return text.replace(' ', '')

def page_scraper(page_name):
    page_text = get_page_contents(page_name)
    if page_text == -1:
        return -1
    
    with open('dataset_new.csv', 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["show #", "airdate","round","category","value", "question","answer"])
        category_list = [] #always 6 long (for single and double, final is just 1 category)

        #gets show # and airdate from the title of the html page
        pos1 = page_text.find("<title>")
        pos2 = page_text.find("</title>")
        title = page_text[pos1:pos2]
        pos1 = title.find("Show #")+6
        show_num = cleaner(title[pos1:pos1+4])
        pos2 = title.find("aired ")+6
        airdate = title[pos2:pos2+10]

        #gets the list of categories for the first round
        pos1 = page_text.find("jeopardy_round")
        page_text = page_text[pos1:]
        
        offset = 0 #we will save all 12 categories to the list; first questions of double jeopardy will then be 6th on list
        round_name = "Jeopardy"

        #multiplies by 2 for double jeopardy round.
        dj_mult=1

        
        for round in range(2):
            
            for i in range(6):
                pos1 = page_text.find("category_name")+len("category_name")+2
                page_text = page_text[pos1:]
                pos2 = page_text.find("</td>")
                #print(page_text[:pos2])
                category_list.append(remove_html(page_text[:pos2]))

            #<td class=\"clue\">\n

            new_mult = 1
            #on November 26 2002 they permanently doubled every question value
            if (int(airdate[:4]) > 2001) or (int(airdate[:4]) == 2001 and int(airdate[5:7])>10 and int(airdate[8:10])>25):
                new_mult = 2
                
            
            
            
            for i in range(30):
                value = (int(i/6)+1)*100*dj_mult*new_mult 
                #print(str(remove_spaces(page_text).find("<tdclass=\"clue\">\n</td>")) +" "+ str(remove_spaces(page_text).find("<tdclass=\"clue\">")))
                if((remove_spaces(page_text).find("<tdclass=\"clue\">\n</td>") == remove_spaces(page_text).find("<tdclass=\"clue\">")) and remove_spaces(page_text).find("<tdclass=\"clue\">\n</td>")>-1):
                    #print("found on "+str(i))
                    question = "[BLANK]"
                    answer = "[BLANK]"
                    page_text = page_text[page_text.find("<td class=\"clue\">"):]
                    page_text = page_text[page_text.find("</td>"):]

                
                
                else:             
                    
                    pos1 = page_text.find("clue_text")+len("clue_text")+2
                    page_text = page_text[pos1:]
                    pos2 = page_text.find("</td>")
                    question = page_text[:pos2]
                    question = remove_html(question)
                       
                    pos1 = page_text.find("correct_response")+len("correct_response")+2
                    page_text = page_text[pos1:]
                    pos2 = page_text.find("</em>")
                    answer = page_text[:pos2]
                    #print(answer)
                    answer = remove_html(answer)
                
                writer.writerow([show_num, airdate, round_name, category_list[(i%6)+offset], value , question, answer])
            if round == 1:
                break
            
            round_name = "Double Jeopardy"
            pos1 = page_text.find("double_jeopardy_round")
            page_text = page_text[pos1:]
            offset = 6
            dj_mult=2

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
        

        value = "Final" 
        
            
        writer.writerow([show_num, airdate, round_name, category_list[12], value , question, answer])
        



#uses page_scraper to scrape multiple pages
def site_scraper(start, end):
    for page_num in range(start, end):
        print(page_num)
        page_scraper("https://j-archive.com/showgame.php?game_id="+str(page_num))
        time.sleep(5)

#each show should have 61 questions; checks where a show starts and isnt divisible by 61 to find an error
def check_errors():
    with open('new_dataset.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        count = 1
        for row in reader:
            if row[2] == 'Final Jeopardy' and not(count%61 == 0):
                print(count)

            count+=1
    
def check_duplicates(fname):
    
    df = pd.read_csv(fname)
    duplicates=df.duplicated(keep=False)
    num = duplicates.sum()
    #print(num)
    df2 = df[duplicates]
    print(df2.head(200))
    
    num_unique = df.drop_duplicates().shape[0]
    print("Number of unique rows:", num_unique)

    print(df.shape)

def check_num_questions(fname):

    dict = {}
    
    with open(fname, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if not( str(row[0]) in dict):
                dict[str(row[0])]=1
            else:
                dict[str(row[0])]+=1
    return dict


#made because there was an error with the new dataset having repeats
def create_final_dataset():
    with open('final_dataset.csv', 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        with open('new_dataset.csv', 'r', encoding='utf-8') as file:
            reader = list(csv.reader(file))
            for i in range(0,182878):
                writer.writerow(reader[i])
            for j in range(282370,len(reader)):
                writer.writerow(reader[j])

#gets rid of all of the unanswered questions
def create_edited_dataset():
    count = 0
    with open('final_dataset_clean.csv', 'a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        with open('final_dataset2.csv', 'r', encoding='utf-8') as file:
            reader = list(csv.reader(file))
            for row in reader:
                if not(row[5]=='[BLANK]'):
                    writer.writerow(row)
                    count+=1
                    print(count)

#in 2001 every question value was doubled; this is to make them the same (first question asked is always 200)
def create_standardized_dataset():
    
    with open('final_dataset_clean_standardized.csv', 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["show #", "airdate","round","category","value", "question","answer"])
        with open('final_dataset_clean.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                airdate = str(row[1])
                if (int(airdate[:4]) > 2001) or (int(airdate[:4]) == 2001 and int(airdate[5:7])>10 and int(airdate[8:10])>25) or row[4] == 'Final':
                    writer.writerow(row)
                else:
                    row[4] = int(row[4])*2
                    writer.writerow(row)
                    
                    

def make_year_df(df, year):
    #splits into a dataframe based on a set of years
    new_df = pd.DataFrame(columns=["show #", "airdate","round","category","value", "question","answer"])
    
    shift = 0
    if(year == 1986):
        #starts it at 1984 (makes the sections more even because there were fewer shows in the first 10 years or so)
        shift = 2 
    
    for i in range(year-shift, year+10):
        temp_df = df[df["airdate"].str.contains(str(i))]
        new_df = pd.concat([new_df,temp_df])

    return new_df
    

def make_value_df(df, value, round): #uses new values starting at $200
    new_df = pd.DataFrame(columns=["show #", "airdate","round","category","value", "question","answer"])
    
    
    temp_df = df[(df["value"].astype(str) == (str(value))) & (df["round"].astype(str) == (str(round)))]
    new_df = pd.concat([new_df,temp_df])

    return new_df
    

if __name__ == '__main__':
    #This file contains various functions that we used throughout the project to create and manipulate the dataset
    
    
    site_scraper(0,100)
    





