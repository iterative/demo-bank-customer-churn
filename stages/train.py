import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from utils.load_params import load_params


def train(models_dir, 
          model_fname, 
          data_dir,
          random_state,
          **train_params):
    X_train = pd.read_pickle(data_dir/'X_train.pkl')
    y_train = pd.read_pickle(data_dir/'y_train.pkl')   
    
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler())
            ]
        )
    categorical_transformer = OrdinalEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    clf = LGBMClassifier(random_state=random_state, 
                                 **train_params)
    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", clf)]
        )
    
    pipe.fit(X_train, y_train)

    models_dir.mkdir(exist_ok=True)
    dump(pipe, models_dir/model_fname)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    params = load_params(params_path=args.config)
    data_dir = Path(params.base.data_dir)
    models_dir = Path(params.train.models_dir)
    model_fname = params.train.model_fname
    random_state = params.base.random_state
    cat_cols = params.base.cat_cols
    num_cols = params.base.num_cols
    train_params = params.train.params

    train(models_dir=models_dir, 
          model_fname=model_fname, 
          data_dir=data_dir,
          random_state=random_state,
          **train_params)
