# Predicting Churn for Bank Customers

## Dataset source
https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers

## Description
This dataset contains 10,000 records, each of it corresponds to a different bank's user. The target is ExitedTask, a binary variable that describes whether the user decided to leave the bank. There are row and customer identifiers, four columns describing personal information about the user (surname, location, gender and age), and some other columns containing information related to the loan (such as credit score, current balance in the user's account and whether they are an active member among others).

## Use Case
The objective is to train a ML model that returns the probability of a customer to churn. This is a binary classification task, therefore F1-score is a good metric to evaluate the performance of this dataset as itweights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously.

