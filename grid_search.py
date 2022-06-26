# see https://dvc.org/blog/hyperparam-tuning
import itertools
import subprocess

# Automated grid search experiments
n_estimators_values = range(50, 250, 50)
max_depth_values = range(10, 25, 5)

# Iterate over all combinations of hyperparameter values.
for n_estimators, max_depth in itertools.product(n_estimators_values, max_depth_values):
    # Execute "dvc exp run --queue --set-param train.n_est=<n_est> --set-param train.min_split=<min_split>".
    subprocess.run(["dvc", "exp", "run", "--queue",
                    "--set-param", f"train.params.n_estimators={n_estimators}",
                    "--set-param", f"train.params.max_depth={max_depth}"])