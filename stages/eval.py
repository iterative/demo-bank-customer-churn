import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dvclive import Live
from eli5.sklearn import PermutationImportance
from joblib import dump
from mlem.api import load
from sklearn.metrics import (confusion_matrix, f1_score, make_scorer,
                             roc_auc_score)
from utils.load_params import load_params


def eval(data_dir, model_dir, perm_imp_model_path, random_state):
    eval_plots_dir = Path('eval_plots')
    eval_plots_dir.mkdir(exist_ok=True)
    live = Live("eval_plots")
    
    X_test = pd.read_pickle(data_dir/'X_test.pkl')
    y_test = pd.read_pickle(data_dir/'y_test.pkl')
    model = load(str(model_dir/"clf-model"))
    y_prob = model.predict_proba(X_test).astype(float)
    y_prob = y_prob[:, 1]
    y_pred = y_prob >= 0.5

    cm = confusion_matrix(y_test, y_pred, normalize='true') 
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
    plt.savefig(eval_plots_dir/'cm.png')
    
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    live.log_sklearn_plot("roc", y_test, y_prob)
    
    preprocessor = model.named_steps['preprocessor']
    clf = model.named_steps['clf']
    X_test_transformed = preprocessor.transform(X_test)

    perm = PermutationImportance(clf, 
                                 scoring=make_scorer(f1_score),
                                 random_state=random_state)
    perm = perm.fit(X_test_transformed, y_test)
    feat_imp = zip(X_test.columns.tolist(), perm.feature_importances_)
    df_feat_imp = pd.DataFrame(feat_imp, 
                      columns=['feature', 'importance'])
    df_feat_imp = df_feat_imp.sort_values(by='importance', ascending=False)
    df_feat_imp.to_csv('feat_imp.csv', index=False)
    dump(perm, perm_imp_model_path)
    
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
    model_dir = Path(params.base.model_dir)
    random_state = params.base.random_state
    perm_imp_model_path = Path(params.eval.perm_imp_model_path)
    eval(data_dir=data_dir, 
         model_dir=model_dir,
         perm_imp_model_path=perm_imp_model_path,
         random_state=random_state)
