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
    
    basket, word_dict  = scraper.create_basket('word_occurences.csv',30,500)

    ans_list, ans_dict  = scraper.create_basket('Answer_occurences.csv',5,20000)

    count = 1
    X_list = []
    y_list = []
    for i in range(0,5):#len(qa_clean)):
        if qa_clean.loc[i,"answer"] in ans_list:
            tempVector = []
            for j in range(0,len(basket)):
                #something fishy here
                if str(basket[j]) in qa.loc[i,"question"]:
                    tempVector.append(1)
                else:
                    tempVector.append(0)
            X_list.append(tempVector)
            y_list.append(ans_list.index(qa_clean.loc[i,"answer"]))
            print(count)
            count+=1
            
    X = np.array(X_list)
    y = np.array(y_list)

    for a in range(0,len(X_list)):
        print("X:", end = " ")
        for b in range(0,len(X_list[a])):
            if X_list[a][b] == 1:
                print(basket[b], end = ", ")
        print("Y: "+str(ans_list[a]))

'''
This is the output:

X: arrest, theory, u, ali, leo, hi, e, g, p, f, ear, ears, le, d, po, o, w, y, rest, al, s, sing, r, h, m, si, ma, un, las, n, t, ho, der, sin, l, il, lil, ye, er, Y: copernicus 
X: 1912, football, seasons, ball, foot, ed, e, p, 19, f, le, d, o, w, y, al, sons, season, s, b, r, h, m, n, t, ho, tar, v, ian, wit, ts, l, seas, di, ba, pi, isle, ants, ol, ant, ave, ia, Y: jim thorpe
X: average, hours, sunshine, suns, u, hi, e, ours, shine, g, f, cord, ear, hour, d, o, y, s, co, r, h, m, ma, un, 55, ate, n, t, ho, rage, ha, v, era, 0, ye, ave, er, Y: arizona 
X: 1963, burger, u, pan, let, 63, ed, hi, ill, e, g, p, 19, le, billion, d, o, w, y, et, s, co, ive, b, r, h, pa, m, ny, iv, n, t, ho, k, v, ts, l, il, lion, 96, serve, ink, er, Y: mcdonalds
X: u, ted, ed, e, sec, g, p, f, d, o, s, co, r, h, m, si, frame, en, ate, resident, n, t, id, con, ram, er, Y: john adams

*** something is wrong with basket
'''
        
    
    #print(np.array(y_list))

    #for ans in y:
        #print(ans_list[ans])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print(y_pred)

    
    
    print("Accuracy:", accuracy_score(y_test, y_pred))

    
    print(datetime.now())    
    print("done")