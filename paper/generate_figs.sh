export PYTHONPATH="${PYTHONPATH}:../"

# -------------------
# --- Figure 1(a) ---
# -------------------
python 1-performance.py --setting regr


# -------------------
# --- Figure 1(b) ---
# -------------------
python 2-time-regression.py --dont_plot --n_params 30 100 300 1000 3000 10000 30000 100000 --n_rep 5 \
   --max_cvxpy 50000
python 2-time-regression.py --load_data --dont_show


# ----------------------
# --- Figure 2 and 3 ---
# ----------------------
for DSET in mnist;
do
  echo $DSET
  for SETTING in compare_lr acceleration stochastic;
  do python 3-convergence-gd.py --setting $SETTING --dset $DSET --dont_plot --n_iter 10000
  done;
done;


# Plot MNIST
python 3-convergence-gd.py --setting compare_lr --dset MNIST --load_data --n_iter 10000 --ls : : : "-"  --dont_show
python 3-convergence-gd.py --setting acceleration --dset MNIST --load_data --n_iter 1000 --dont_show
python 3-convergence-gd.py --setting  stochastic --dset MNIST --load_data --n_iter 100 --xlabel "\#  epochs" --dont_show
# Plot breast cancer
python 3-convergence-gd.py --setting compare_lr  --dset breast_cancer  --n_iter 1000 --ls : : : "-" --dont_show
python 3-convergence-gd.py --setting acceleration --dset breast_cancer  --n_iter 100 --dont_show
python 3-convergence-gd.py --setting stochastic --dset breast_cancer --n_iter 100 --xlabel "\#  epochs" --dont_show

# --------------------------
# --- Figure 4 top + S.2 ---
# --------------------------
python 4-convergence-regression.py --dset abalone


python 4-convergence-regression.py --dset wine --n_iter 30
python 4-convergence-regression.py --dset diabetes --n_iter 100

# -----------------------
# --- Figure 4 bottom ---
# -----------------------
# now running on Hyperion (screen  253869.time-eval)
python 5-syntetic-dsets.py --setting comparing_advtrain_linf_methods --dont_plot --n_points 30 --increasing_scale 1000
python 5-syntetic-dsets.py --setting comparing_advtrain_linf_methods --plot_type time --load_data

# ----------------------------
# --- Figure 5  ---
# ----------------------------
for SETTING in spiked_covariance sparse_gaussian;
  do python 5-syntetic-dsets.py  --setting $SETTING  --dont_show
done;

# ------------------
# --- Figure S.1 ---
# ------------------


# ------------------
# --- Figure S.2 ---
# ------------------
python 1-performance.py --setting classif


# ------------------
# --- Figure S.3 ---
# ------------------
# now running on Hyperion (screen  253869.time-eval)
python 5-syntetic-dsets.py --setting comparing_advtrain_linf_methods_classif --dont_plot --n_points 10 --increasing_scale 50
python 5-syntetic-dsets.py --setting comparing_advtrain_linf_methods_classif --plot_type time --load_data




# -----------------------------
# --- Neurips Rebutal Fig 4 ---
# -----------------------------
python 3-convergence-gd.py --setting batch_size --dset breast_cancer --n_iter 1000 --dont_plot
python 3-convergence-gd.py --setting batch_size --dset breast_cancer --n_iter 500 --load_data --dont_plot