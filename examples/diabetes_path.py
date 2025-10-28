from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn import datasets
from sklearn import linear_model
import tqdm
from linadvtrain.regression import lin_advregr
from linadvtrain.regression import get_radius


def get_lasso_path(X, y, eps_lasso=1e-5):
    alphas, coefs, _ = lasso_path(X, y, eps=eps_lasso)
    coefs= np.concatenate([np.zeros([X.shape[1], 1]), coefs], axis=1)
    alphas = np.concatenate([1e2 * np.ones([1]), alphas], axis=0)
    return alphas, coefs, []


def get_path(X, y, estimator, amax, eps=1e-5, n_alphas=200):
    amin = eps * amax
    alphas = np.logspace(np.log10(amin), np.log10(amax), n_alphas)
    coefs_ = []
    info_ = []
    for a in tqdm.tqdm(alphas):
        coefs, info = estimator(X, y, a)
        coefs_.append(coefs if coefs is not None else np.zeros(m))
        info_.append(info)
    return alphas, np.stack((coefs_)).T, info_


def plot_coefs(alphas, coefs, ax):
    colors = cycle(["b", "r", "g", "c", "k"])
    for coef_l, c in zip(coefs, colors):
        ax.semilogx(1/alphas, coef_l, c=c)


def diabetes_path():
    X, y = datasets.load_diabetes(return_X_y=True)
    # Standardize data
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    fig, ax = plt.subplots(num='lasso')
    alphas_lasso, coefs_lasso, _ = get_lasso_path(X, y)
    plot_coefs(alphas_lasso, coefs_lasso, ax)

    fig, ax = plt.subplots(num='ridge')
    estimator = lambda X, y, a: (linear_model.Ridge(alpha=a, fit_intercept=False).fit(X, y).coef_, {})
    alphas_ridge, coefs_ridge, _ = get_path(X, y, estimator, 1e4)
    plot_coefs(alphas_ridge, coefs_ridge, ax)

    fig, ax = plt.subplots(num='advtrain_l2')
    estimator = lambda X, y, a:  lin_advregr(X, y, adv_radius=a, p=2)
    alphas_adv, coefs_advtrain_l2, info_l2 = get_path(X, y, estimator, 1e1)
    plot_coefs(alphas_adv, coefs_advtrain_l2, ax)
    ax.plot(1/get_radius(X, y, 'zero', p=2), 0, 'o', ms=10, color='black')
    ax.axvline(1/get_radius(X, y, 'randn_zero', p=2))

    fig, ax = plt.subplots(num='advtrain_linf')
    estimator = lambda X, y, a:  lin_advregr(X, y, adv_radius=a, p=np.inf)
    alphas_adv, coefs_advtrain_linf, info_linf = get_path(X, y, estimator, 1e1)
    plot_coefs(alphas_adv, coefs_advtrain_linf, ax)
    ax.plot(1/get_radius(X, y, 'zero', p=np.inf), 0, 'o', ms=10, color='black')
    ax.axvline(1/get_radius(X, y, 'randn_zero', p=np.inf))


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('webagg')
    diabetes_path()
    plt.show()

