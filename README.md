# FinalProject_2025_LucasTilly
This is my implementation of a Recommender System Project using the Kuairec Dataset

# Objective
Having the big_matrix as a train test and the small_matrix as a test set created by forcing user to watch certain videos, I decided that I was going to try and order all videos seen by a user in small_matrix by predicting their watch ratio. That way, the objective is to be able to recommend the best videos for each users.

# Approach
To choose the model I was going to use, I looked into the classe's notebooks, and realized that a two-towers model would probably match this project the most.
The plan was to have one tower as the video features, and the other as the user features.
However, when I tried this method giving one-hot encoded features to each tower, I ended up with huge matrices that my computer's memory couldn't handle.
Therefore, taking those limitations into account, I did some feature engineering to be able to reduce the number of dimensions. This new approach having less features and with features connecting both users and videos, a two-towers model wasn't the best fit anymore, so I decided to go with a single-tower model.

# How to run this project
Use python 3.11.12 and retrieve packages from requirements.txt
Run each jupyter notebook one by one in the indicated order