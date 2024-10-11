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
# --- Figure 2 + S.1 ---
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



# -----------------------------
# --- Fig 3(top) ---
# -----------------------------
# Top (Deterministic)
python 3-convergence-gd.py --setting fgsm --dset breast_cancer --n_iter 1000 --dont_plot
python 3-convergence-gd.py --setting fgsm --dset breast_cancer --n_iter 500 --load_data --dont_plot
# Bottom (Stochastic)
python 3-convergence-gd.py --setting fgsm-sgd --dset breast_cancer --n_iter 100 --dont_plot


# --------------------------
# --- Figure 4 top + S.2 ---
# --------------------------
python 4-convergence-regression.py --dset abalone
python 4-convergence-regression.py --dset wine --n_iter 30
python 4-convergence-regression.py --dset diabetes --n_iter 100

# -----------------------
# --- Figure 4 bottom ---
# -----------------------
python 5-syntetic-dsets.py --setting comparing_advtrain_linf_methods --dont_plot --n_points 30 --increasing_scale 1000
python 5-syntetic-dsets.py --setting comparing_advtrain_linf_methods --plot_type time --load_data

# ----------------------------
# --- Figure 5  ---
# ----------------------------
for SETTING in spiked_covariance sparse_gaussian;
  do python 5-syntetic-dsets.py  --setting $SETTING  --dont_show
done;



# -----------------------------
# --- Fig S.3 ---
# -----------------------------
python 3-convergence-gd.py --setting batch_size --dset breast_cancer --n_iter 1000 --dont_plot
python 3-convergence-gd.py --setting batch_size --dset breast_cancer --n_iter 500 --load_data --dont_plot



# ------------------
# --- Figure S.5 ---
# ------------------=
python 5-syntetic-dsets.py --setting comparing_advtrain_linf_methods_classif --dont_plot --n_points 10 --increasing_scale 50
python 5-syntetic-dsets.py --setting comparing_advtrain_linf_methods_classif --plot_type time --load_data

# ------------------
# --- Figure S.6 ---
# ------------------
python 1-performance.py --setting classif



# -----------------------------
# --- Table S.1---
# -----------------------------
python comparing_cvxpy_configs.py


# -----------------------------
# --- Extra ---
# -----------------------------
python 5-syntetic-dsets.py --setting magic --n_reps 5 --dont_plot
python 5-syntetic-dsets.py --setting magic --n_reps 5 --load_data

