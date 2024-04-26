import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso, Ridge, RidgeCV, LogisticRegression
from linadvtrain.classification import lin_advclasif
from datasets import *
import matplotlib as mpl
import seaborn as sns
import time
import matplotlib.pyplot as plt
from linadvtrain.regression import lin_advregr
from  sklearn import ensemble, neural_network

from sklearn.metrics import (r2_score, root_mean_squared_error, mean_absolute_percentage_error, roc_auc_score,
                             average_precision_score)

# Basic style
plt.style.use(['mystyle.mpl'])


# Additional style
mpl.rcParams['figure.figsize'] = 7, 3
mpl.rcParams['figure.subplot.bottom'] = 0.15
mpl.rcParams['figure.subplot.right'] = 0.99
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 22
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.3
mpl.rcParams['xtick.major.pad'] = 7

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --------------------------
# --- Regression Methods ---
# --------------------------
def lasso_cv(X_train, y_train):
    regr = ElasticNetCV(cv=10, random_state=0, l1_ratio=1)
    regr.fit(X_train, y_train)
    return regr.coef_

def advtrain_linf(X_train, y_train):
    estimated_params, info = lin_advregr(X_train, y_train, p=np.inf)
    return estimated_params


def get_quantiles(xaxis, r, quantileslower=0.25, quantilesupper=0.75):
    new_xaxis, inverse, counts = np.unique(xaxis, return_inverse=True, return_counts=True)

    r_values = np.zeros([len(new_xaxis), max(counts)])
    secondindex = np.zeros(len(new_xaxis), dtype=int)
    for n in range(len(xaxis)):
        i = inverse[n]
        j = secondindex[i]
        r_values[i, j] = r[n]
        secondindex[i] += 1
    m = np.median(r_values, axis=1)
    lerr = m - np.quantile(r_values, quantileslower, axis=1)
    uerr = np.quantile(r_values, quantilesupper, axis=1) - m
    return new_xaxis, m, lerr, uerr


def plot_errorbar(df, xname, yname, ax, label, color='blue'):
    x = df[xname].values
    y = df[yname].values
    new_x, m, lerr, uerr = get_quantiles(x, y)
    ax.errorbar(new_x, m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                marker='o', markersize=3.5, ls='', label=label, color=color)



if __name__ == "__main__":
    import pandas as pd
    n_reps = 5
    n_train_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    all_methods = [advtrain_linf, lasso_cv]
    rng = np.random.RandomState(1)
    n_test=500
    all_results = {m.__name__: np.zeros(n_reps * len(n_train_list)) for m in all_methods}
    all_results['n_train'] = np.zeros(n_reps * len(n_train_list))
    i = 0
    for n_train in n_train_list:
        for rep in range(5):
            n_params = int(0.1*n_train)
            X_train = rng.randn(n_train, n_params)
            X_test = rng.randn(n_test, n_params)
            beta = rng.randn(n_params)
            y_train = X_train @ beta + 1 * rng.randn(n_train)
            y_test = X_test @ beta + 1 * rng.randn(n_test)
            n_train, n_params = X_train.shape
            # Test dimension
            for method in all_methods:
                params = method(X_train, y_train)
                y_pred = X_test @ params
                all_results[method.__name__][i] = root_mean_squared_error(y_test, y_pred)
                all_results['n_train'][i] = n_train
            i += 1


    df = pd.DataFrame(all_results)

    fig, ax = plt.subplots()
    plot_errorbar(df, 'n_train', 'advtrain_linf', ax, 'adv train')
    plot_errorbar(df, 'n_train', 'lasso_cv', ax, 'lasso_cv', color= 'red')
    plt.xlabel('RMSE')
    plt.xlabel('\# samples')
    plt.legend()
    plt.show()



