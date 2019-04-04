# Troll Tweet Classification

## Objective
The objective of this project is to detect whether a tweet was sent from an account tied to Russia's Internet Research Agency based purely on its text content. The scope of the project is looking at tweets specifically sent during the highest activity of these accounts discovered by the House Intelligence Committee investigation, identified in the dataset reconstructed by NBC News and available at https://www.kaggle.com/vikasg/russian-troll-tweets. These messages were compared against tweets collected on Harvard's Dataverse for the 2016 Presidential election at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDI7IN. Samples were taken from the 'Candidates and key election hashtags' collection and their content was compared to that from the Kaggle set in order to verify that their topics were similar to justify comparison. The time frame was narrowed to those in common between the two sets.


## Data Sources:
* Non-Troll Tweets
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDI7IN
These tweets are contained in a collection and must be rehydrated into jsonl files in order to recreate the project in the TrollTweetClassification Jupyter Notebook. Note that our chosen collection has several million tweets. For the scope of our project we chose to sample every 400th tweet from the collection based on our resources and time constraints. The ids for the samples we chose to rehydrate are located in the non-troll-tweets folder as txt files. Information on how to hydrate the tweets is located in the link above (we used twarc). Note that the jsonl files created totaled approximately 2.5 GB.
* Troll Tweets
https://www.kaggle.com/vikasg/russian-troll-tweets/
These tweets and the related user information are stored in csv files in the  folder russian-troll-tweets as 'tweets.csv' and 'users.csv'. 

## Data Cleaning, Analysis and Model Evaluation
The process of cleaning, analyzing and creating the models based on this corpus is located in the TrollTweetClassification Jupyter Notebook and can be recreated once the above data has been regathered. 

## Conclusions
Our final model created by stacking two separate Stochastic Gradient Descent Classifiers managed to identify 49-57% of Troll tweets while falsely flagging about 12-16% of regular tweets, depending on the chosen weight parameter. The AUC for the final model reached .77 for the unweighted case which was impressive given the few features fed into the model. Next steps for the project would include further analysis and treatment of the hashtags and mentions contained in the tweets as well as using more advanced Natural Language Processing tools like sentiment analysis and language detection. Based on our exploration in dealing with the class imbalance further improvements in the performance of the model are likely with oversampling instead of undersampling.