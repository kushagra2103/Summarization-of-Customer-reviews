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

### LSTM Cell

![1](https://user-images.githubusercontent.com/36281158/91410373-71b0b080-e864-11ea-91ad-a78d92581f75.PNG)

Here 'h' represents the hidden state, 'c' represents the cell state, 't' represents the  current time stamp. 

![2](https://user-images.githubusercontent.com/36281158/91412893-010b9300-e868-11ea-9043-c39356a37e31.PNG)

The LSTM has three parts; forget gate, update gate and output gate. 

#### How the cell works ?

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation. The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

##### Forget Gate 

![3](https://user-images.githubusercontent.com/36281158/91521094-8187cd80-e914-11ea-85ab-28c0521bda50.PNG)

This gate is primarly used for information that needs to be forget. If the context changes, for example if there is gender change (male to female) or from plural to singular or vice versa, the cell needs to remeber that context has changed. Forget gate plays a role by "not transferring" the male/ plural information to the cell state. It uses the sigmoid function which outputs 0 or 1. 0 means completely forget and 1 means retains the whole information. Wf is  a matrix which is combination of two weight s matrices for xt and ht-1. bf is the bias added. 

##### Update Gate 

![4](https://user-images.githubusercontent.com/36281158/91527106-b6028600-e922-11ea-804b-beb5253ee410.PNG)

It has an input layer that decides what values we need to update (like from male to female/ plural to singular). So it is passed again to a neural layer network same as forget layer which ouputs 0/1 and then is multiplied by cell state c~ which will keep the new information and stored in the cell state. Now the previous cell state ct-1 and this current updated c~ are added which will form the current cell state ct which will again passed as ct-1 for the next time step.

![5](https://user-images.githubusercontent.com/36281158/91527133-c581cf00-e922-11ea-8748-ac53f7c9e7f6.PNG)

##### Output Gate 

![6](https://user-images.githubusercontent.com/36281158/91530085-19db7d80-e928-11ea-98ab-ec53415439c7.PNG)

Now to decide what goes to output, a output sigmoid function is again defined same as forget gate. It is responsible to pass the information that a context has changed and it might output the information relevant to the "new" context. So multiplied by tanh(ct) and is output as ht. For final output, a dense layer is applied to each time step/ last time step depending upon whether it is many to many / many to one structure.  


### Encoder-Decoder Sequential Model 

![8](https://user-images.githubusercontent.com/36281158/91533315-5067c700-e92d-11ea-83e8-c2aeafa04026.png)

#### Summary of the above model 

1. Both the encoder and decoder have LSTM cells as units

2. Encoder reads the input text in form of time steps, embedding layer converts it into the word embeddings and all the information in defined number of time steps are passed as initial vectors for decoder; c(cell state) and h (hidden state)

3. Decoder model then predicts the outputs at every time step , where the output at previous time step is feed as an input to the next time step to predict the next output.

4. Output at every time step from decoder model is then combined and the resultant summary is formed


### Text Cleaning and Preprocessing 

Before the data is split into training and testing, our data contains a lot of noise which are needed to remove in order for our model to predict better.

1. Missing data 

There are some missing data in both Review Text and Title, they needed to be removed, I have used "dropna" function which basically removes all the rows which has atleast one missing data 

2. Noise 

There are noise in form of special characters, capital/small letters, punctuations, numbers, short forms like in place of 'is not' it is 'isn't', one letter words, stop_words etc which are needed to be removed in order to improve the quality of the data our model will feed upon.

![12](https://user-images.githubusercontent.com/36281158/91565820-68a50980-e960-11ea-9ca8-4d43ce86efe2.PNG)

here the text is cleaned part.

3. Defining the fix length for input to encoder and output for decoder; in short defining the time steps for encoder and decoder. Seeing the histograms, we can fix the input time steps to be 50 and for output to be 7. It means each example will have max length of 50, if length is < 50, it will be "post paded" and if length > 50 then it will be truncated, same for output. So summary will have max length of 7, it can be< 7 also. 

![13](https://user-images.githubusercontent.com/36281158/91566344-229c7580-e961-11ea-8425-400ec29a5518.PNG)











































