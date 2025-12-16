import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import scraper 
from sklearn.preprocessing import LabelEncoder
#from datasets import Dataset
from transformers import AutoTokenizer
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict, concatenate_datasets, ClassLabel

'''
Somethings not in the ml environment might need to be installed
'''


#getting data ready, selecting classes (most common answers)
def create_df_filtered():
    df = scraper.our_dataframe()
    qa = df[["question", "answer","airdate","value","round"]]
    qa_clean = qa.applymap(scraper.cleaner) 
    
    top_answers = qa_clean['answer'].value_counts().head(300).index.tolist()
    
    df_filtered = qa_clean[qa_clean['answer'].isin(top_answers)].copy()
    
    le = LabelEncoder()
    df_filtered['label'] = le.fit_transform(df_filtered['answer'])
    
    print("Most commonly occuring answers: ", le.classes_)
    
    
    
    our_dataset = Dataset.from_pandas(df_filtered[['question', 'label']])
    
    return df_filtered

#splits df_filtered into the 4 time frames
def create_years_list(df_filtered):
    df_years_list = []
    for i in range(0,4):
        df_years_list.append(scraper.make_year_df(df_filtered,1986+(10*i)))
    return df_years_list

#makes a 2d array which includes questions of certain values for each time period
def create_vals_list(df_years_list):
    df_vals_list = []
    
    for year in range(0,4):
    
        temp = []
        round = "jeopardy" #round must be lowercase because we ran cleaner already; cause me a lot of frustration 
        dj_mult=1 #for double jeopardy
        for val in range(0,10):
            if (val>4):
                dj_mult=2
                round = "double jeopardy"
            df_val= scraper.make_value_df(df_years_list[year],200*dj_mult*((val%5)+1),round)
            
            print(str(200*dj_mult*((val%5)+1))+' '+str(df_val.shape[0]))

            #need to make sure labels are right and that we trim down into proper train and test before plopping it in the list
            
            
            new_dataset = Dataset.from_pandas(df_val[['question', 'label']])
            
            hfDataset = new_dataset.train_test_split(test_size=0.2, seed=67)
            print("size of val: " +str(hfDataset["train"].num_rows))
            
            #df_val_split = [df_val, hfDataset["train"], hfDataset["test"]]
            #df_val_split = [df_val, hfDataset]
            #temp.append(df_val_split)
            temp.append(hfDataset)
    
        
        df_vals_list.append(temp)

    return(df_vals_list)
    
#we make splits of train in test data based on value so they'll be pretty even and then remake the year train and test data
def create_year_dataset_dict(year_range, df_vals_list): 

    train_set = df_vals_list[year_range][0]["train"]
    test_set = df_vals_list[year_range][0]["test"]
    for i in range(1,10):
        train_set = concatenate_datasets([train_set, df_vals_list[year_range][i]["train"]])
        test_set = concatenate_datasets([test_set, df_vals_list[year_range][i]["test"]])

    ds_dict = DatasetDict({
        "train": train_set, 
        "test": test_set
    })
    return ds_dict
    
#we make splits of train in test data based on value so they'll be pretty even and then remake the year train and test data
def create_full_vals_dataset_dict(val, df_vals_list): 

    train_set = df_vals_list[0][val]["train"]
    test_set = df_vals_list[0][val]["test"]
    for i in range(1,4):
        train_set = concatenate_datasets([train_set, df_vals_list[i][val]["train"]])
        test_set = concatenate_datasets([test_set, df_vals_list[i][val]["test"]])

    ds_dict = DatasetDict({
        "train": train_set, 
        "test": test_set
    })
    
    return ds_dict


def create_full_dataset_dict(dict_list):


    train_set = dict_list[0]["train"]
    test_set = dict_list[0]["test"]
    for i in range(1,4):
        train_set = concatenate_datasets([train_set, dict_list[i]["train"]])
        test_set = concatenate_datasets([test_set, dict_list[i]["test"]])

    final_dict = DatasetDict({
        "train": train_set, 
        "test": test_set
    })

    return final_dict


        



if __name__ == '__main__':

    #ended up having to put the contents of df_filtered in the main method because of errors with scope
    df = scraper.our_dataframe()
    qa = df[["question", "answer","airdate","value","round"]]
    qa_clean = qa.applymap(scraper.cleaner) 
    
    top_answers = qa_clean['answer'].value_counts().head(300).index.tolist()
    
    df_filtered = qa_clean[qa_clean['answer'].isin(top_answers)].copy()
    
    le = LabelEncoder()
    df_filtered['label'] = le.fit_transform(df_filtered['answer'])
    
    print("Most commonly occuring answers: ", le.classes_)
    
    
    
    
    print(df_filtered.shape[0])

    #splits into 4 years
    df_years_list=create_years_list(df_filtered)

    #splits 4 years into 10 categories
    df_vals_list=create_vals_list(df_years_list)

    
    
    for i in range(0,10):
        print(df_vals_list[0][i]["train"].num_rows)

    #we then reconstruct the dataset, first into 4 parts...
    dict_list = []
    for a in range(0,4):
        new = create_year_dataset_dict(a,df_vals_list)
        dict_list.append(new)

    #... and then the 4 parts become one
    final = create_full_dataset_dict(dict_list)
    #we now have both a large dataset containing everything and the subdatasets, which we can use later to test

    final = final.cast_column("label", ClassLabel(num_classes=300))

    
    print(final)
    
    
    
    
    dataset = final
    
    print(dataset)
    
    
    
    model_checkpoint = "distilbert-base-uncased"


    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    
    #preprocessing
    
    def preprocess_function(examples):
        return tokenizer(examples["question"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    
    
    accuracy = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    
    to_label = {}
    to_id = {}
    
    for index, label_name in enumerate(le.classes_):
        to_label[index] = label_name
        to_id[label_name] = index
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=300,
        id2label=to_label,
        label2id=to_id
    )
    
    
    #bert finetuning here
    #experiment with different parameters
    #definitely need to mess around with these
    args = TrainingArguments(
        output_dir="bert_final",
        eval_strategy="steps",    
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("training")
    trainer.train()
    
    #print(trainer.evaluate())
    
    model.save_pretrained("bert_final")
    tokenizer.save_pretrained("bert_final")
    