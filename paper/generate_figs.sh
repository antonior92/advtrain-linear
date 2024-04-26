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
for DSET in magic_classif;
do
  echo $DSET
  for SETTING in compare_lr acceleration stochastic;
  do python 3-convergence-gd.py --setting $SETTING --dset $DSET --dont_plot
  done;
done;

for DSET in  mnist breast_cancer magic_classif;
do
  echo $DSET
  for SETTING in compare_lr acceleration stochastic;
  do python 3-convergence-gd.py --setting $SETTING --dset $DSET --dont_show --load_data
  done;
done;

# ------------------
# --- Figure S.1 ---
# ------------------


# ------------------
# --- Figure S.2 ---
# ------------------
python 1-performance.py --setting classif