import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso, Ridge, RidgeCV, LogisticRegression
from linadvtrain.classification import lin_advclasif
from datasets import *
import matplotlib as mpl
import time
import matplotlib.pyplot as plt
from linadvtrain.regression import lin_advregr, get_radius
from  sklearn import ensemble, neural_network
from linadvtrain.adversarial_attack import compute_adv_attack

from sklearn.metrics import (r2_score)

# Basic style
plt.style.use(['mystyle.mpl'])

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


all_methods = [advtrain_linf, lasso_cv]
mylabels = ['Adv Train', 'Lasso CV']

if __name__ == '__main__':
    n_reps = 10
    n_train = 60
    n_test = 100
    n_params = 40

    configs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    all_results = {'R-squared': np.zeros(n_reps * len(configs) * len(all_methods)),
                   'R-squared-adv': np.zeros(n_reps * len(configs) * len(all_methods)),
                   'alpha': np.zeros(n_reps * len(configs)* len(all_methods)),
                   'time': np.zeros(n_reps * len(configs)* len(all_methods)),
                   'methods': [''] * n_reps * len(configs) * len(all_methods)}
    i = 0

    for rep in range(n_reps):

        X_train, X_test, y_train, y_test = latent_features(n_train, n_test, n_test, seed=1, n_latent=1, noise_std=0.1)
        for n, method in zip(mylabels, all_methods):
            print(n)
            start_time = time.time()
            params = method(X_train, y_train)
            exec_time = time.time() - start_time
            y_pred = X_test @ params
            for alpha in configs:
                # evaluate adversarial train
                Dx_test = compute_adv_attack(y_pred - y_test, params, ord=np.Inf)
                y_pred_adv = (X_test + alpha * Dx_test) @ params
                all_results['R-squared'][i] = r2_score(y_test, y_pred)
                all_results['R-squared-adv'][i] = r2_score(y_test, y_pred_adv)
                all_results['time'][i] = exec_time
                all_results['alpha'][i] = alpha
                all_results['methods'][i] = n
                i += 1
    df = pd.DataFrame(all_results)



    fig, ax = plt.subplots()
    c = ['red', 'blue', 'green', 'cyan']
    for i, m in enumerate(all_methods):
        plot_errorbar(df[df['methods'] == mylabels[i]], 'alpha',  'R-squared-adv', ax, mylabels[i], color=c[i])
    plt.ylabel('R-squared Adv')
    plt.xlabel('adv. radius')
    plt.xscale('log')
    plt.legend()
    plt.savefig('imgs/rebuttal.pdf')
    plt.show()