import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
from dvclive import Live
from joblib import load
from sklearn.metrics import f1_score, plot_confusion_matrix, roc_auc_score
from utils.load_params import load_params


def eval(models_dir, 
         model_fname, 
         data_dir):
    eval_plots_dir = Path('eval_plots')
    eval_plots_dir.mkdir(exist_ok=True)
    live = Live("eval_plots")
    
    X_test = pd.read_pickle(data_dir/'X_test.pkl')
    y_test = pd.read_pickle(data_dir/'y_test.pkl')
    model = load(models_dir/model_fname)
    y_prob = model.predict_proba(X_test)
    y_prob = y_prob[:, 1]
    y_pred = y_prob >= 0.5

    plot_confusion_matrix(model, X_test, y_test, normalize='true', cmap=plt.cm.Blues)  
    plt.savefig(eval_plots_dir/'cm.png')
    
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    live.log_plot("roc", y_test, y_prob)
        
    df_test = X_test
    df_test['true'] = y_test
    df_test['pred'] = y_pred
    df_test['prob'] = y_prob

    metrics = {
        'f1': f1,
        'roc_auc': roc_auc
    }
    json.dump(
        obj=metrics,
        fp=open('metrics.json', 'w'),
        indent=4, 
        sort_keys=True
    )

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    params = load_params(params_path=args.config)
   
    data_dir = Path(params.base.data_dir)
    models_dir = Path(params.train.models_dir)
    model_fname = params.train.model_fname

    eval(models_dir=models_dir, 
         model_fname=model_fname, 
         data_dir=data_dir)
