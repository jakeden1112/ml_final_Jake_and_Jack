import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import scraper
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split




if __name__ == '__main__':

    print(datetime.now())

    df = scraper.hf_dataframe()
    
    qa = df[["question","answer"]]
    
    qa_clean = qa.map(scraper.cleaner)
        
    #occ_dict = scraper.occ_dict_maker(qa_clean,'question')
    
    basket, word_dict  = scraper.create_basket('word_occurences.csv',20,1000)

    ans_list, ans_dict  = scraper.create_basket('Answer_occurences.csv',10,20000)

    #scraper.find_good_questions(qa_clean, ans_list, 'good_questions.csv')

    good_qs = scraper.retrieve_good_qs('good_questions.csv')

    #print(good_qs)
    
    #print(datetime.now())
    count = 1
    X_list = []
    y_list = []
    
    #print (len(basket))
    #print(len(ans_list))

    num_qs = 20000
    
    #goes through each question and makes an X vector using one hot encoding and a Y vector, which has a number represent the answer
    for i in good_qs[0:num_qs]:
        
        tempVector = np.zeros(len(basket)).tolist()
        for word in qa_clean.loc[int(i),"question"].split():
            #print(word)
            if word in basket:
                tempVector[basket.index(word)] = 1
                
        X_list.append(tempVector)
        y_list.append(ans_list.index(qa_clean.loc[int(i),"answer"]))
        #print(count)
        #count+=1
       
    X = np.array(X_list)
    y = np.array(y_list)
    
    #for ans in y:
        #print(ans_list[ans])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k = 1
    
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    #print(y_pred)
    
    '''
    for a in range(0,len(X_list)):
        print("X:", end = " ")
        for b in range(0,len(X_list[a])):
            if X_list[a][b] == 1:
                print(basket[b], end = ", ")
        print("Y: "+str(ans_list[a]))
    '''
    print("using "+str(num_qs)+" samples and a k of " +str(k))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    
    print(datetime.now())    
    print("done")
    