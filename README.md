# Summarization-of-Customer-reviews

## Problem Statement

Here we will see how the entire context of a particular text can be automatically generated in a precise less number of words. I have used the kaggle data set (https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews/home) for this case. I have used RNN (LSTM ENCODER - DECODER) structure to generate the summaries of the reviews. All the work is done in Jupyter notebook, conda environment; python 3.7 with keras in backend. The purpose of this project is to learn how RNN (LSTM) SeqtoSeq model works and this type of problem can be extrapolated to many other problems let it be document summarization in finance, healthcare. 

## Dataset Description

The dataset is Women Ecommerce clothing reviews. 

This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review, and includes the variables:

Clothing ID: Integer Categorical variable that refers to the specific piece being reviewed.

Age: Positive Integer variable of the reviewers age.

Title: String variable for the title of the review.

Review Text: String variable for the review body.

Rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.

Recommended IND: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.

Positive Feedback Count: Positive Integer documenting the number of other customers who found this review positive.

Division Name: Categorical name of the product high level division.

Department Name: Categorical name of the product department name.

Class Name: Categorical name of the product class name.

#### We will be using the Title and reviews Text for summarization.

## Theory Behind the Encoder-Decoder Sequential Model 














































