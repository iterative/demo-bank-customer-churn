import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import f1_score, roc_auc_score
from utils.load_params import load_params


def eval(metrics_fpath,
         models_dir, 
         model_fname, 
         data_dir):
    X_test = pd.read_pickle(data_dir/'X_test.pkl')
    y_test = pd.read_pickle(data_dir/'y_test.pkl')
    clf = load(models_dir/model_fname)
    y_prob = clf.predict_proba(X_test)
    y_pred = y_prob[:, 1] >= 0.5
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    
    df_test = X_test
    df_test['true'] = y_test
    df_test['pred'] = y_pred
    df_test['prob'] = y_prob[:, 1]
    
    f1_by_geo = df_test.groupby('Geography').apply(lambda x: f1_score(x['true'], x['pred'])).to_dict()
    roc_auc_by_geo = df_test.groupby('Geography').apply(lambda x: roc_auc_score(x['true'], x['prob'])).to_dict()
    
    metrics = {
        'f1': f1,
        'roc_auc': roc_auc,
        'f1_by_geo': f1_by_geo,
        'roc_auc_by_geo': roc_auc_by_geo
    }
    json.dump(
        obj=metrics,
        fp=open(metrics_fpath, 'w'),
        indent=4, 
        sort_keys=True
    )

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    params = load_params(params_path=args.config)
    metrics_fpath = Path(params.eval.metrics_fpath)
    data_dir = Path(params.base.data_dir)
    models_dir = Path(params.train.models_dir)
    model_fname = params.train.model_fname

    eval(metrics_fpath=metrics_fpath,
         models_dir=models_dir, 
         model_fname=model_fname, 
         data_dir=data_dir)
