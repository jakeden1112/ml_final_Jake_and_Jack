# ml_final_Jake_and_Jack



## Overview
Brief (roughly one paragraph) overview of your project, including its aims and the main findings/outcome (at a high level)

The final aim of our project was to examine whether or not jeopardy questions have changed in difficulty over time. For the first two phases of our project, we used an existing jeopardy question set from hugging face. We ran and trained a distilbert model on questions containing an answer from the 20 most common answers. However, this only translated to approximately 3000 entries. For our final phase, we wanted a better dataset and to train our distilbert model on a much larger set of questions. We made our own jeopardy question dataset, which was achieved by scraping a jeopardy archive site. Out of this massive dataset, we decided on using questions containing answers in the 300 most common. This translated to a little under 50,000 entries. We split these entries into 4 ~10 year time periods, and wanted to see what period the distilbert model would have the highest accuracy on. To run the distilbert model on a set this large, we had to use Colgate's supercomputer services. We tweaked the model's settings until we found what we thought to be an optimal number of epochs, which was 4. For each time period, the model achieved signifcantly better than random chance (1/300). We found that the model was able to perform at slightly over 50% accuracy for the first, second, and fourth time periods and a little under 50% accuracy for the third time period. Since these results were relatively consistent, we did not conclude that the difficulty of jeopardy questions has changed over time, but rather stayed relatively similar. 

## Replication Instructions

In order to replicate the results we have in our poster, you would need to use our final dataset, 'final_dataset_clean.csv'
- This dataset was built by scraping j-archive.com
- You would then need to create a new dataframe containing only questions containing an answer in the 300 most common
- From this dataframe, you would separate questions based on year. There were 4 ~10 year time periods we used: 1984-1995, 1996-2005, 2006-2015, 2016-2025.
- Once you have the entries separated into these categories, you want to train our distilbert model with the provided settings using 4 epochs. (as a note, we needed to connect to Colgate's supercomputer to run our distilbert model)
- Then you test the trained distilbert model on the testing data, using accuracy as the metric
  
## Future Directions

Brief (roughly one paragraph) overview of next steps/ways to improve on/concrete extensions of your project

I think that we could expand the model to have a certainty metric in each question that it answers. This could be a probability between 0 and 1, which we could achieve through the use of a softmaxing function. There are also other categories that we could examine to see if they have any effect on the performance of the model. These could be question category, and the prevalence of a questions answer across jeopardy as a whole. Since we focused on training the model solely on questions with answers that were very prevalent in the dataset, we could see how the model performs on questions with answers that were much less prevalent. This would decrease the model's accuracy by a lot, and probably be out of the scope of the class to build a model that can answer questions like this, but nevertheless, it would be very interesting to try to build a model that can answer niche questions. 

## Contributions
Details on the contributions of each member to the project, including time spent (if a group of one you should still do this)

Jack
- Built initial BERT model and ran it to get results for phase 2
- Data analysis on final question set for context in our poster/presentation
- Worked on poster

  A significant amount of time was spent building and running our first BERT model. Around 7-8 hours (across several sessions). Also spent ~2 hours on data analysis. Work on poster probably totaled 1-2 hours. 

Jake
## add


## everything below this point is from earlier stages of the project, we can probably delete it


## Project roadmap going forward

-Find existing NLP classification models that will allow us to sort questions and answers into predefined categories. 
-apply different models (possibly BERT), make sure to cite sources of models 
-Be able to have model answer SOME questions.

meeting notes
-online tutorials for fine tuning bert
- 

## Goal for first milestone

Note: We will refer to what the host reads as the question and what the contestant says in response as the answer to avoid confusion

We want to start off by getting a feel for manipulating the data and then applying a simple model on it to see if we can get any sort of result. We are using a dataset of Jeopardy quesions from Huggingface to start off, although in the future, we plan to scrape our own dataset, since on the website J Archive there is listed over 500,000 Jeopardy questions, while the Huggingface dataset has less than 300,000 entries. 

## Rationale

Rather than just throw all of the data from the dataset into a model and see what happens, we were thoughtful in which questions from the dataset we used and how they are represented. We decided to use one hot encoding with whether or not each word is present being a feature. We had to make some choices to not have very large vectors. We took out words that are super common because they won't add much information and they might confuse the model. 
For example: consider this question and answer Q:'The section of this river near London Bridge is called the Pool' A:'The Thames' We should not have the words 'the,' 'of,' 'this,' 'is' being given the same weight (or any weight at all) as 'river,' 'bridge,' 'London,' 'pool.' We also got rid of words that are super uncommon, since if a word only shows up in one or two questions, it won't be very useful for predicting answers to unseen questions, since it is unlikely to show up again and it doesn't affect the training very much. We have also taken out answers that only appear a few times because if there aren't enough instances of them, the model won't get much of a sense of them while training and will have a hard time knowing what to look for if it shows up when we test the model.

## What we did

We made two files, scraper.py and model.py. scraper.py will eventually include scraping methods, but for now we have just included methods to clean a dataset, since we are using hugging face for now. model.py is being used to turn the data into X vectors and y values and then putting them into a model.

## Summary of functions

#### hf_dataframe():
Returns a dataframe of the huggingface dataset

#### cleaner(text):
Takes text and cleans it. Uses regular expression to remove any non alphanumeric symbols, makes lower case, and tries to remove all instances of multiple spaces

#### occ_dict_maker (df, column):
Makes a dictionary of unique words used in questions and the amount of times they occur. 

#### answer_dict_maker(df, column):
Gives us the occurences of different answers. Unlike occ_dict_maker, it puts the whole string into the dictionary. The idea is that we want to count how many answers are repaeated. If an answer is only used once or a few times, than we shouldn't train it on it because it is unlikely to be seen again. We want to only train on answers that are given somewhat often so that the model will be able to have enough instances to learn off of.

#### dict_to_csv (occ_dict, fname):
Saves the dictionary to a csv file that we can easily read off of to avoid the computational cost of making dictionaries every time.

#### create_basket (fname, low_bound, up_bound):
This creates our basket of words. We are currently using one hot encoding to have an binary representation for each word if it is present or not in a question. The list basket_list is a representation of the mapping of which words will be repersented by which index of the basket in a vector. output_dict is just fetching the dictionary of occurences that we had previously stored. We use the upper and lower bounds as culling the words down so we don't have words that will throw the model off. We also have it remove words that are less than 3 letters, as these are unlikely to contain much information (we might change this, since it was mainly added to account for a problem which was fixed)

#### find_good_questions(df, ans_list, fname):
This is just to make a csv file that will have the questions that we want to use (where the answer is well represented in the dataset), so we don't have to check if a question has an answer is in the answer set each time we make a vector

    
#### retrieve_good_qs(fname):
Gets a list of questions we will use out of the csv file
    
## What we are doing in model.py

We load in the Huggingface dataset and then make qa, which is a dataframe of just the questions and answers. We then clean qa out using the function in scraper.py. Then we load in all of the lists and dictionaries we'll need. We make two lists that will be lists of lists of for our X vectors and y values. We loop through the questions we have decided are good to use, marking 1 in a vector if a certain word shows up and then putting the vectors in a list of X vectors and their associated y values in a list. We then turn these lists of lists into numpy arrays that we can then split into training and test data. We then train knn with the training data and test it with the test data to get an accuracy.

## How does it look?

We have been tweaking the amount of data used as well as the k value. We have been getting accuracy levels that are pretty low, but that is to be expected when we are using such a simple model on such a hard problem. 

As expected, increasing the k value decreases the accuracy. Jeopardy questions and answers are very imbalanced/unique. 

using 20000 samples and a k of 1 Accuracy: 0.0375 
using 20000 samples and a k of 2 Accuracy: 0.02075
using 20000 samples and a k of 3 Accuracy: 0.01975

## What's next?

We would like to try running on Turing using a job since using too many vectors can cause the computer to crash. We want to try other models that are more advanced, like neual nets. We want to scrape our own larger dataset from the J Archive website and use more advanced cleaning techniques. We have also been thinking about our methods of choosing which words to keep in the representation. We have thought about using a dictionary and maybe prioritizing certain words (such as proper nouns).

