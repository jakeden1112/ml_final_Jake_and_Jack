import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import scraper 
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification



#getting data ready, selecting classes (most common answers)
df = scraper.our_dataframe()
qa = df[["question", "answer"]]
qa_clean = qa.applymap(scraper.cleaner) 

top_answers = qa_clean['answer'].value_counts().head(400).index.tolist()

df_filtered = qa_clean[qa_clean['answer'].isin(top_answers)].copy()

le = LabelEncoder()
df_filtered['label'] = le.fit_transform(df_filtered['answer'])

print("Most commonly occuring answers: ", le.classes_)



our_dataset = Dataset.from_pandas(df_filtered[['question', 'label']])

dataset = our_dataset.train_test_split(test_size=0.2)

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
    num_labels=400,
    id2label=to_label,
    label2id=to_id
)


#bert finetuning here
#experiment with different parameters
#definitely need to mess around with these
args = TrainingArguments(
    output_dir="bert_jeopardy_model_400",
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