import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from utils.load_params import load_params


def train(models_dir, 
          model_fname, 
          data_dir,
          random_state):
    X_train = pd.read_pickle(data_dir/'X_train.pkl')
    y_train = pd.read_pickle(data_dir/'y_train.pkl')
    clf = LGBMClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    models_dir.mkdir(exist_ok=True)
    dump(clf, models_dir/model_fname)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    params = load_params(params_path=args.config)
    data_dir = Path(params.base.data_dir)
    models_dir = Path(params.train.models_dir)
    model_fname = params.train.model_fname
    random_state = params.base.random_state

    train(models_dir=models_dir, 
          model_fname=model_fname, 
          data_dir=data_dir,
          random_state=random_state)
