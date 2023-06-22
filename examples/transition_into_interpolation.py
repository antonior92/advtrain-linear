#%%
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn import datasets
from sklearn import linear_model
import tqdm
from linadvtrain.solvers import lin_advtrain, get_radius
import matplotlib
matplotlib.use('webagg')

#%%
# Path lengh
eps_lasso = 1e-5
eps_ridge = 1e-6
eps_adv = 1e-5
# alpha_max
amax_ridge = 1e4
amax_adv = 10
# number of points along the path
n_alphas = 40
# alpha_min path (automatically computed)
amin_ridge = eps_ridge * amax_ridge
amin_adv = eps_adv * amax_adv

# Generate Gaussian dataset
parameter_norm = 1
noise_std = 0.1
n_features = 200
n_samples = 60
seed = 1


#%%
rng = np.random.RandomState(seed)
beta = 1 / np.sqrt(n_features) * np.ones(n_features)
X = rng.randn(n_samples, n_features)

# Generate output with random additive noise
e = rng.randn(n_samples)
y = X @ beta + noise_std * e

#%%
ridge = lambda X, y, a: linear_model.Ridge(alpha=a, fit_intercept=False).fit(X, y).coef_
advtrain_l2 = lambda X, y, a: lin_advtrain(X, y, adv_radius=a, p=2)[0]
advtrain_linf = lambda X, y, a: lin_advtrain(X, y, adv_radius=a, p=np.inf)[0]

def compute_coefs_path(estimator, amin, amax, n_alphas):
    alphas = np.logspace(np.log10(amin), np.log10(amax), n_alphas)
    coefs_ = []
    for a in tqdm.tqdm(alphas):
        coefs_.append(estimator(X, y, a))
    return alphas, np.stack((coefs_)).T

# ridge
alphas_ridge, coefs_ridge = compute_coefs_path(ridge, amin_ridge, amax_ridge, n_alphas)
# lasso
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, n_alphas=n_alphas, eps=eps_lasso, fit_intercept=False)
coefs_lasso = np.concatenate([np.zeros([X.shape[1], 1]), coefs_lasso], axis=1)
alphas_lasso = np.concatenate([1e2 * np.ones([1]), alphas_lasso], axis=0)
# adv train l2
alpha_advtrain_l2 , coefs_advtrain_l2  = compute_coefs_path(advtrain_l2, amin_adv, amax_adv, n_alphas)
# adv train linf
alpha_advtrain_linf , coefs_advtrain_linf  = compute_coefs_path(advtrain_linf, amin_adv, amax_adv, n_alphas)

#%% Plot MSE

def plot_mse(alphas, coefs, name):
    fig, ax = plt.subplots(num=name)
    mse = np.mean((y[:, None] - X @ coefs) **2, axis=0)
    ax.plot(1/alphas, mse, 'o-', label='Ridge')
    if name in ['advtrain_l2' , 'advtrain_linf']:
        p = np.inf if name == 'advtrain_linf' else 2
        ax.axvline(1 / get_radius(X, y, 'interp', p=p))
    ax.set_xscale('log')
    ax.set_yscale('log')

def transition_into_interpolation():
    plot_mse(alphas_ridge, coefs_ridge, 'ridge')
    plot_mse(alphas_lasso, coefs_lasso, 'lasso')
    plot_mse(alpha_advtrain_l2, coefs_advtrain_l2, 'advtrain_l2')
    plot_mse(alpha_advtrain_linf, coefs_advtrain_linf, 'advtrain_linf')

if __name__ == "__main__":
    transition_into_interpolation()
    plt.show()