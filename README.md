# Predicting Churn for Bank Customers

## Dataset source
https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers

## Description
This dataset contains 10,000 records, each of it corresponds to a different bank's user. The target is ExitedTask, a binary variable that describes whether the user decided to leave the bank. There are row and customer identifiers, four columns describing personal information about the user (surname, location, gender and age), and some other columns containing information related to the loan (such as credit score, current balance in the user's account and whether they are an active member among others).

## Use Case
The objective is to train a ML model that returns the probability of a customer to churn. This is a binary classification task, therefore F1-score is a good metric to evaluate the performance of this dataset as itweights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously.


## Installation
Python 3.8+ is required to run code from this repo.
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

1. Create and configure a location for [remote storage](https://dvc.org/doc/command-reference/remote/add#supported-storage-types) (e.g. AWS S3) OR setup a local [DVC remote](https://dvc.org/doc/command-reference/remote#example-add-a-default-local-remote).
Change the pointer to your DVC remote in `.dvc/config`.
2. Download `Churn_Modeling.csv` file from [here](https://www.kaggle.com/datasets/santoshd3/bank-customers?select=Churn+Modeling.csv) and place it in `data/Churn_Modelling.csv`

Now you can start a Jupyter Notebook server and execute the notebook `notebook/TrainChurnModel.ipynb` top to bottom to train a model

```bash
$ jupyter notebook
```

If you want to run the DVC pipeline:
```bash
dvc repro # runs the pipeline defined in `dvc.yaml`
dvc push # pushes the resulting artifacts to a DVC remote configured in `.dvc/config`
```
