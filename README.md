# Predicting Churn for Bank Customers

## Dataset source
https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers

## Description
This dataset contains 10,000 records, each of it corresponds to a different bank's user. The target is ExitedTask, a binary variable that describes whether the user decided to leave the bank. There are row and customer identifiers, four columns describing personal information about the user (surname, location, gender and age), and some other columns containing information related to the loan (such as credit score, current balance in the user's account and whether they are an active member among others).

|RowNumber|CustomerId|Surname |CreditScore|Geography|Gender|Age|Tenure|Balance |NumOfProducts|HasCrCard|IsActiveMember|EstimatedSalary|Exited|
|---------|----------|--------|-----------|---------|------|---|------|--------|-------------|---------|--------------|---------------|------|
|1        |15634602  |Hargrave|619        |France   |Female|42 |2     |0       |1            |1        |1             |101348.88      |1     |
|2        |15647311  |Hill    |608        |Spain    |Female|41 |1     |83807.86|1            |0        |1             |112542.58      |0     |
|3        |15619304  |Onio    |502        |France   |Female|42 |8     |159660.8|3            |1        |0             |113931.57      |1     |
|4        |15701354  |Boni    |699        |France   |Female|39 |1     |0       |2            |0        |0             |93826.63       |0     |
...

## Use Case
The objective is to train a ML model that returns the probability of a customer to churn. This is a binary classification task, therefore F1-score is a good metric to evaluate the performance of this dataset as itweights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously.

![](https://miro.medium.com/max/456/1*Dvx1j18vyKyvLlIpxzVSmQ.png)

## What's DVC? 

DVC is an open-source tool for data management and ML experiment management designed to integrate well with git and CI/CD tools.

DVC is available for [Visual Studio Code](https://dvc.org/doc/vs-code-extension), on any [system terminal](https://dvc.org/doc/install), and as a [Python library](https://dvc.org/doc/api-reference).

![](https://ucarecdn.com/d11a1937-b684-4410-a7d1-d24c074fae86/)

## Core DVC features
### Data versioning
![](https://editor.analyticsvidhya.com/uploads/86351git-dvc.png)
### ML pipelines
![](https://martinfowler.com/articles/cd4ml/ml-pipeline-2.png)
### Experiment management
![](https://cdn.thenewstack.io/media/2022/04/2b0eb28b-mm3.png)

## Project Setup
Python 3.8+ is required to run code from this repo.
```bash
$ git clone https://github.com/iterative/demo-bank-customer-churn
$ cd demo-bank-customer-churn
```

Now let's install the requirements. But before we do that, we strongly recommend
 creating a virtual environment with a tool such as `virtualenv`:

```bash
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Setup DVC remote

If you'd like to test commands like [`dvc push`](https://man.dvc.org/push), that require write access to the remote storage, the easiest way would be to set up a "local remote" on your file system.
This kind of remote is located in the local file system, but is external to the DVC project.
You'll also need to download the dataset from https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers
and place it into the `data/` directory.
```bash
$ mkdir -p /tmp/dvc-storage
$ dvc remote add local /tmp/dvc-storage
```
You should now be able to run:
```bash
$ dvc push -r local
```

## Running DVC pipeline

Set `PYTHONPATH` to the project's path:
```bash
$ export PYTHONPATH=$PWD
```
Execute DVC pipeline
```bash
$ dvc repro
```


