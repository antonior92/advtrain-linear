#%%
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn import datasets
from sklearn import linear_model
import tqdm
from linadvtrain.solvers import lin_advtrain
import matplotlib
matplotlib.use('webagg')


#%%

# Path lengh
eps_lasso = 1e-5
eps_ridge = 1e-6
eps_adv = 1e-5
# alpha_max
amax_ridge = 1e4
amax_adv = 1
# number of points along the path
n_alphas = 200
# alpha_min path (automatically computed)
amin_ridge = eps_ridge * amax_ridge
amin_adv = eps_adv * amax_adv

X, y = datasets.load_diabetes(return_X_y=True)
n, m = X.shape
# Standardize data (easier to set the l1_ratio parameter)
X -= X.mean(axis=0)
X /= X.std(axis=0)

# Compute lasso paths
print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=eps_lasso)
coefs_lasso = np.concatenate([np.zeros([X.shape[1], 1]), coefs_lasso], axis=1)
alphas_lasso = np.concatenate([1e2 * np.ones([1]), alphas_lasso], axis=0)

# Compute ridge paths
alphas_ridge = np.logspace(np.log10(amin_ridge), np.log10(amax_ridge), n_alphas)
coefs_ridge_ = []
for a in tqdm.tqdm(alphas_ridge):
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs_ridge_.append(ridge.coef_)
coefs_ridge = np.stack((coefs_ridge_)).T

alphas_adv = np.logspace(np.log10(amin_adv), np.log10(amax_adv), n_alphas)
coefs_advtrain_l2_ = []
coefs_advtrain_linf_ = []
info_l2 = []
info_linf = []
for a in tqdm.tqdm(alphas_adv):
    coefs, info = lin_advtrain(X, y, adv_radius=a, p=2)
    coefs_advtrain_l2_.append(coefs if coefs is not None else np.zeros(m))
    info_l2.append(info)
    coefs, info = lin_advtrain(X, y,  adv_radius=a, p=np.inf, method='w-ridge')
    info_linf.append(info)
    coefs_advtrain_linf_.append(coefs if coefs is not None else np.zeros(m))  # p = infty seems ill conditioned
coefs_advtrain_l2 = np.stack((coefs_advtrain_l2_)).T
coefs_advtrain_linf = np.stack((coefs_advtrain_linf_)).T


#%%
# Display results

def plot_coefs(alphas, coefs, name):
    fig, ax = plt.subplots(num=name)

    colors = cycle(["b", "r", "g", "c", "k"])
    for coef_l, c in zip(coefs, colors):
        ax.semilogx(1/alphas, coef_l, c=c)
    if name == 'advtrain_linf':
        axt = ax.twinx()
        axt.semilogx(1/alphas, [info['n_iter'] for info in info_linf])

    if name == 'advtrain_l2':
        axt = ax.twinx()
        axt.semilogx(1 / alphas, [info['n_iter'] for info in info_l2])



plot_coefs(alphas_lasso, coefs_lasso, 'lasso')
plot_coefs(alphas_ridge, coefs_ridge, 'ridge')
plot_coefs(alphas_adv, coefs_advtrain_l2, 'advtrain_l2')
plot_coefs(alphas_adv, coefs_advtrain_linf, 'advtrain_linf')


if __name__ == '__main__':
    plt.show()

