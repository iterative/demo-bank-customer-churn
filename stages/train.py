import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

import pandas as pd
from lightgbm import LGBMClassifier
from mlem.api import save
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from utils.load_params import load_params
from xgboost import XGBClassifier
import gto
import dvc.api
import joblib


def train(data_dir,
          model_dir,
          model_type,
          cat_cols,
          random_state,
          update_model:str=None,
          **train_params):
    X_train = pd.read_pickle(data_dir/'X_train.pkl')
    y_train = pd.read_pickle(data_dir/'y_train.pkl')

    if model_type == "randomforest":
        clf = RandomForestClassifier(random_state=random_state, 
                                 **train_params)
    elif model_type == "lightgbm":
        clf = LGBMClassifier(random_state=random_state, 
                                    **train_params)
    elif model_type == "xgboost":
        clf = XGBClassifier(random_state=random_state, 
                                    **train_params)

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
    if update_model:
        repo = "."
        revision = gto.api.find_versions_in_stage(repo=repo, name=update_model, stage="prod")[0].ref
        with dvc.api.open("model/clf-model", repo=repo, rev=revision, mode="rb") as f:
            model = joblib.load(f)
            model["clf"].n_estimators += (train_params.get("n_estimators") - model["clf"].n_estimators)
    else:
        model = Pipeline(
            steps=[("preprocessor", preprocessor), ("clf", clf)]
            )
    
    model.fit(X_train, y_train)
    save(
        model,
        model_dir / "clf-model",
        sample_data=X_train
    )

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    params = load_params(params_path=args.config)
    data_dir = Path(params.base.data_dir)
    model_dir = Path(params.base.model_dir)
    random_state = params.base.random_state
    cat_cols = params.base.cat_cols
    num_cols = params.base.num_cols
    model_type = params.train.model_type
    train_params = params.train.params
    update_model = params.train.update_model
    train(data_dir=data_dir,
          model_dir=model_dir,
          model_type=model_type,
          cat_cols=cat_cols,
          random_state=random_state,
          update_model=update_model,
          **train_params)
