# Predicting Churn for Bank Customers

## Dataset source
https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers

## Description
This dataset contains 10,000 records, each of it corresponds to a different bank's user. The target is ExitedTask, a binary variable that describes whether the user decided to leave the bank. There are row and customer identifiers, four columns describing personal information about the user (surname, location, gender and age), and some other columns containing information related to the loan (such as credit score, current balance in the user's account and whether they are an active member among others).

## Use Case
The objective is to train a ML model that returns the probability of a customer to churn. This is a binary classification task, therefore F1-score is a good metric to evaluate the performance of this dataset as itweights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously.


## Installation
Python 3.7+ is required to run code from this repo.
```bash
$ git clone https://github.com/iterative/demo-bank-customer-churn
$ cd demo-bank-customer-churn
```

Now let's install the requirements. But before we do that, we strongly recommend
 creating a virtual environment with a tool such as `virtualenv`:

```bash
$ virtualenv -p python3 .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Running in your environment

This DVC project comes with a preconfigured DVC 
[remote storage](https://dvc.org/doc/command-reference/remote) that holds raw 
data (input), intermediate, and final results that are produced. 
This is a read-only HTTP remote.

```bash
$ dvc remote list
storage https://remote.dvc.org/demo-bank-customer-churn
```

You can run dvc pull to download the data:
```bash
$ dvc pull
```

Now you can start a Jupyter Notebook server and execute the notebook `notebook/TrainChurnModel.ipynb` top to bottom to train a model

```bash
$ jupyter notebook
```

If you'd like to test commands like [`dvc push`](https://man.dvc.org/push), that require write access to the remote storage, the easiest way would be to set up a "local remote" on your file system:

This kind of remote is located in the local file system, but is external to the DVC project.
```bash
$ mkdir -p /tmp/dvc-storage
$ dvc remote add local /tmp/dvc-storage
```
You should now be able to run:
```bash
$ dvc push -r local
```
