declare -a n_estimators_values=(50 100 150)
declare -a max_depth_values=(10 15 20)
## now loop through the above array
for n_est in "${n_estimators_values[@]}"
    do
        for max_d in "${max_depth_values[@]}"
            do
            dvc exp run -S train.params.n_estimators=$n_est -S train.params.max_depth=$max_d
            done
    done