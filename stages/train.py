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


def train(n_estimators,
          num_leaves,
          learning_rate,
          max_depth,
          reg_alpha,
          reg_lambda,
          models_dir, 
          model_fname, 
          data_dir,
          cat_cols,
          num_cols,
          random_state):
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
                         n_estimators=n_estimators,
                         num_leaves=num_leaves,
                         max_depth=max_depth,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         learning_rate=learning_rate)
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
    n_estimators = params.train.n_estimators
    num_leaves = params.train.num_leaves
    learning_rate = params.train.learning_rate
    max_depth = params.train.max_depth
    reg_alpha = params.train.reg_alpha
    reg_lambda = params.train.reg_lambda
    model_fname = params.train.model_fname
    random_state = params.base.random_state
    cat_cols = params.base.cat_cols
    num_cols = params.base.num_cols
    
    train(n_estimators=n_estimators,
          num_leaves=num_leaves,
          learning_rate=learning_rate,
          max_depth=max_depth,
          reg_alpha=reg_alpha,
          reg_lambda=reg_lambda,
          models_dir=models_dir, 
          model_fname=model_fname, 
          cat_cols=cat_cols,
          num_cols=num_cols,
          data_dir=data_dir,
          random_state=random_state)
