# ml_final_Jake_and_Jack



## Overview

The final aim of our project was to examine whether or not jeopardy questions have changed in difficulty over time. For the first two phases of our project, we used an existing jeopardy question set from hugging face. We ran and trained a distilbert model on questions containing an answer from the 20 most common answers. However, this only translated to approximately 3000 entries. For our final phase, we wanted a better dataset and to train our distilbert model on a much larger set of questions. We made our own jeopardy question dataset, which was achieved by scraping the website j-archive.com. Out of this massive dataset, we decided on using questions containing answers in the 300 most common, as these almost all contained at least 100 each. This translated to a little under 50,000 entries. We split these entries into 4 ~10 year time periods, and wanted to see what period the distilbert model would have the highest accuracy on. We tweaked the model's settings until we found what we thought to be an optimal number of epochs, which was 4. For each time period, the model achieved significantly better than random chance (1/300). We found that the model was able to perform at close to 50% accuracy for the four time periods. We determined that accuracy would give us a proper measure of difficulty, as we observed a noticeable downward trend in accuracy as question value increased. The show is structured so that questions of greater dollar value are more difficult, so this correlation makes sense. We then averaged out the questions over the time periods and measured the accuracy of each. While the 4 time periods could be said to show a slight decreasing trend in accuracy, we were not convinced that it was enough of a trend to say that the questions had truly become more difficult overall. If we interpret it that the questions have stayed relatively similar in difficulty, this is a good outcome, as it shows the writing on the show has been consistent.

## Replication Instructions

In order to replicate the results we have in our poster, you would need to use our final dataset, 'final_dataset_clean_standardized.csv'
Note: We used the nlp environment in turing to run some of the programs, so they might not work as well in the ml environment

### To build the dataset

- Use the functions in scraper.py to scrape j-archive.com.
- You can use the site_scraper function to enter the starting and ending page you want.
    - A call to site scraper loops through all of the given pages and sends each page number to page_scraper which then scrapes the page with that number. Every page has the same address except followed by a different number which serves as the id of the show. site_scraper waits a little before going on to the next page to avoid overwhelming j-archive's servers.
- Take note of which csv file the function is writing to and if you don't want to have to start over each time change the 'w' in the write function to 'a' for append.
    - We didn't make the dataset all in one go, so we had to append as to not lose what we had already scraped.
- page_scraper can run on most pages, as they all have pretty much the same structure of html. However, there were a few pages that were giving us trouble for which we had to go into the function and make temporary changes.
    - For example, one head scratcher of an issue involved an exception being thrown after page_scraper called the remove_html function. What turned out to be happening was the question contained a '<' character, so when it was put into the remove_html function, it didnt see the '<' matched with a '>' so it went into an infinite loop, since the remove_html function finds the '<' and '>' of an html tag and removes them and everything inside of them, granted each '<' is paired with a '>'. To fix this, we could have made remove_html more robust or taken the html junk into the csv and then run the function on the whole dataset later.
- When we made the dataset, we had the program running in many separate instances, some locally and others remotely through turing. This meant that multiple csv files were made and then manually concatenated. In theory, you could make a similar dataset just by running it once, though, barring any hiccups caused by broken pages or strange formatting.
- We used create_edited_dataset to take out all of the blank questions.
    - If a question was not read on the show due to time running out, they would not save it to j-archive as no one has any record as to what it was. We saved these questions as the page_scraper function treats every page the same (as in there are 61 questions always). While the blank questions might be helpful for a dataset used to keep record of questions, it is not helpful at all for training a model. 
- We used create_standardized_dataset to double older dollar values too keep things standard
    - In 2001 every question was doubled. Before then, the first question of the first round would be 100 and the second 200. After they changed it in 2001, the same levels of questions would be 200 and 400 respectively. We did this so we could analyze questions based on dollar value without having to account for this change. There shouldn't be a difference in difficulty after the dollar values were doubled, it was just done to keep up with economic factors.
 
### Training the model

- Training was done with run_bert.py, testing was done with test_bert.py
- First you create the df_filtered dataframe, which contains the 300 most commonly occurring questions. You get this by taking the top 300, cleaning them using the cleaner function from scraper (removes any punctuation, makes all lowercase, etc.) and then making the answers into labels that BERT can interpret.
- You can then separate based on time period using create_years_list and separate each time period into values using create_vals_list
    - There were 4 ~10 year time periods we used: 1984-1995, 1996-2005, 2006-2015, 2016-2025. We extended the first range to account for the fact that there were fewer shows in the first few years. In retrospect, this split might have not been the best as it made the whole dataset roughly even, but when turned into the top 300, it might have got a little too unbalanced. This might have affected accuracy of the test data, and contributed to us questioning the slightly higher accuracy of the earlier years.
    - We wanted to split the dataset into these groups before training to make sure the train test split had roughly the same number from each year range and each value within each year range. 
- You can then work backwards, combining the values dataset into datasets for the 4 year ranges and then those 4 into one big dataset using create_year_dataset_dict and create_full_vals_dataset_dict. You can then use the new concatenated train split to train the dataset.
    - This method might have not been the best, but it allowed us to get pretty evenly distributed datasets for testing that we wouldn't have gotten had we split the original df_filtered into train and test and then split up the test, as the original split probably wouldn't have split up the other factors evenly.
- We use the large train dataset to train a DistilBERT model for 4 epochs.
- Then you test the trained distilbert model on the various subdatasets we made, using accuracy as the metric
  
## Future Directions


It would be good to make further changes to control for external variable that might be affecting the outcome of the experiment. For example, even using the top 300, the most prevalent answers showed up much more than the least prevalent with the top showing up around 500 and the bottom around 100. Capping them all at around 100 would make the model perform worse, but it would make the experiment more controlled. In terms of extending the project, we could expand the model to have a certainty metric in each question that it answers. This could be a probability between 0 and 1, which we could achieve through the use of a softmaxing function. We could then use this to measure the models performance on a simulated jeopardy show, where it only buzzes in to questions that pass a certain level of certainty. Like a contestant on the show, the model would not need to know the answer to every question to do well, rather it would just have to know when it will be right and when it will be wrong to end up netting a positive sum. There are also other factors that we could examine to see if they have any effect on the performance of the model. These could be question category, and the prevalence of a question's answer across jeopardy as a whole. Since we focused on training the model solely on questions with answers that were very prevalent in the dataset, we could see how the model performs on questions with answers that were much less prevalent. This would decrease the model's accuracy by a lot, but we could help to improve it by creating new questions for training based on lesser represented questions.

## Contributions
Details on the contributions of each member to the project, including time spent (if a group of one you should still do this)

Jack
- Built initial BERT model and ran it to get results for phase 2
- Data analysis on final question set for context in our poster/presentation
- Worked on poster

  A significant amount of time was spent building and running our first BERT model. Around 7-8 hours (across several sessions). Also spent ~2 hours on data analysis. Work on poster probably totaled 1-2 hours. 

Jake
- Wrote scraper functions to compile dataset and ran them
- Wrote functions to turn dataset into train and test data for BERT model
- Made graphs based on accuracy of test data
- Worked on poster

 Spent way too much time tweaking functions and compiling dataset, maybe around 10-12 hours across multiple days (minus the time waiting for the scraping to happen), although a lot of it was spent debugging, double checking, and redoing things that got messed up, so it shouldn't have taken so long. Spent maybe 5-6 hours working on getting dataset split up properly and maybe 1-2 hours making graphs. Poster probably tooke 1-2 hours. I don't know if these times are accurate. I spent most of the time being a dum dum staring at error messages.
 


