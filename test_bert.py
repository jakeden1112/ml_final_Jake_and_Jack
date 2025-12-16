import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import scraper 
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict, concatenate_datasets, ClassLabel
import run_bert
import seaborn as sns
import matplotlib.pyplot as plt

'''
I mostly ran this file in turing using the nlp environment, so I don't know how well it will work with the ml environment

This file was mainly to make graphs
There were a lot of changes made to it, so I might have used if for things that are no longer apart of it
I have left some stuff that should make some graphs and whatnot, but not all of them are accounted for
'''


model = AutoModelForSequenceClassification.from_pretrained("bert_final")


df = scraper.our_dataframe()
qa = df[["question", "answer","airdate","value","round"]]
qa_clean = qa.applymap(scraper.cleaner) 

top_answers = qa_clean['answer'].value_counts().head(300).index.tolist()

df_filtered = qa_clean[qa_clean['answer'].isin(top_answers)].copy()

le = LabelEncoder()
df_filtered['label'] = le.fit_transform(df_filtered['answer'])

print("Most commonly occuring answers: ", le.classes_)





print(df_filtered.shape[0])

df_years_list=run_bert.create_years_list(df_filtered)
df_vals_list=run_bert.create_vals_list(df_years_list)




for i in range(0,10):
    print(df_vals_list[0][i]["train"].num_rows)

dict_list = []
for a in range(0,4):
    new = run_bert.create_year_dataset_dict(a,df_vals_list)
    dict_list.append(new)

final = run_bert.create_full_dataset_dict(dict_list)


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

ds1 = dict_list[0].map(preprocess_function, batched=True)
ds2 = dict_list[1].map(preprocess_function, batched=True)
ds3 = dict_list[2].map(preprocess_function, batched=True)
ds4 = dict_list[3].map(preprocess_function, batched=True)

dict_list2 = []
for b in range(0,10):
    new = run_bert.create_full_vals_dataset_dict(b,df_vals_list)
    dict_list2.append(new)

dsv_list = []
for i in range(0,10):
    dsv = dict_list2[i].map(preprocess_function, batched=True)
    dsv_list.append(dsv)

'''
dsv1= dict_list2[0].map(preprocess_function, batched=True)
dsv2 = dict_list2[1].map(preprocess_function, batched=True)
dsv3 = dict_list2[2].map(preprocess_function, batched=True)
dsv4 = dict_list2[3].map(preprocess_function, batched=True)
dsv5 = dict_list2[4].map(preprocess_function, batched=True)
dsv6 = dict_list2[5].map(preprocess_function, batched=True)
dsv7 = dict_list2[6].map(preprocess_function, batched=True)
dsv8 = dict_list2[7].map(preprocess_function, batched=True)
dsv9 = dict_list2[8].map(preprocess_function, batched=True)
dsv10 = dict_list2[9].map(preprocess_function, batched=True)
'''





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
    num_train_epochs=1,
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


ds1 = ds1.cast_column("label", ClassLabel(num_classes=300))
prediction1 = trainer.predict(ds1["test"])
acc1 = prediction1.metrics["test_accuracy"]

ds2 = ds2.cast_column("label", ClassLabel(num_classes=300))
prediction2 = trainer.predict(ds2["test"])
acc2 = prediction2.metrics["test_accuracy"]

ds3 = ds3.cast_column("label", ClassLabel(num_classes=300))
prediction3 = trainer.predict(ds3["test"])
acc3 = prediction3.metrics["test_accuracy"]

ds4 = ds4.cast_column("label", ClassLabel(num_classes=300))
prediction4 = trainer.predict(ds4["test"])
acc4 = prediction4.metrics["test_accuracy"]


accuracies = [acc1, acc2, acc3, acc4]
datasets = ["1984-1995", "1996-2005", "2006-2015", "2016-2025"]

plt.bar(datasets, accuracies)
plt.ylabel("Accuracy")
plt.title("Model's accuracy across years")
plt.savefig("f1_years_bar.png")

#makes a list of all the accuracy measures for questions of each dollar value
predictions_list = []
acc_list = []
for i in range(0,10):
    dsv_list[i] = dsv_list[i].cast_column("label", ClassLabel(num_classes=300))
    prediction = trainer.predict((dsv_list[i])["test"])
    predictions_list.append(prediction)
    acc = predictions_list[i].metrics["test_accuracy"]
    acc_list.append(acc)

accuracies1 = [acc_list[0], acc_list[1], acc_list[2], acc_list[3], acc_list[4]]
datasets1 = ["$200", "$400", "$600", "$800", "$1000"]

accuracies2 = [acc_list[5], acc_list[6], acc_list[7], acc_list[8], acc_list[9]]
datasets2 = ["$400", "$800", "$1200", "$1600", "$2000"]

#two graphs are made for jeopardy and double jeopardy round
plt.bar(datasets1, accuracies1)
plt.ylabel("Accuracy")
plt.title("Accuracy Across Dollar Values (Jeopardy Round)")
plt.savefig("acc_vals_bar_j.png")

plt.bar(datasets2, accuracies2)
plt.ylabel("Accuracy")
plt.title("Accuracy Across Dollar Values (Double Jeopardy Round)")
plt.savefig("acc_vals_bar_dj.png")

#makes graphs of accuracy for particular dollar value over years
#make changes to second number to have different values
dsv1 = df_vals_list[0][0].map(preprocess_function, batched=True)
dsv2 = df_vals_list[1][0].map(preprocess_function, batched=True)
dsv3 = df_vals_list[2][0].map(preprocess_function, batched=True)
dsv4 = df_vals_list[3][0].map(preprocess_function, batched=True)

dsv1 = dsv1.cast_column("label", ClassLabel(num_classes=300))
prediction1 = trainer.predict(dsv1["test"])
a1 = prediction1.metrics["test_accuracy"]

dsv2 = dsv2.cast_column("label", ClassLabel(num_classes=300))
prediction2 = trainer.predict(dsv2["test"])
a2 = prediction2.metrics["test_accuracy"]

dsv3 = dsv3.cast_column("label", ClassLabel(num_classes=300))
prediction3 = trainer.predict(dsv3["test"])
a3 = prediction3.metrics["test_accuracy"]

dsv4 = dsv4.cast_column("label", ClassLabel(num_classes=300))
prediction4 = trainer.predict(dsv4["test"])
a4 = prediction4.metrics["test_accuracy"]

accuracies3 = [a1, a2, a3, a4]
datasets3 = ["1984-1995", "1996-2005", "2006-2015", "2016-2025"]

#change out description and filename to get new graph
#to make graphs, commented out rest and changed numbers up abouve and text below each time
plt.bar(datasets3, accuracies3)
plt.ylabel("Accuracy")
plt.title("$200 (1st round) Questions Across Time")
plt.savefig("acc_400(2).png")


print("done")
