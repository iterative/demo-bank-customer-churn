#!/bin/sh
dvc exp run -S 'train.params.n_estimators=range(50, 200, 30)' -S 'train.params.max_depth=range(10, 25, 5)' --queue