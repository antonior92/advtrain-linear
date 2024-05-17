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

from sklearn.metrics import (r2_score)

# Basic style
plt.style.use(['mystyle.mpl'])


# Additional style
mpl.rcParams['figure.figsize'] = 7, 3
mpl.rcParams['figure.subplot.bottom'] = 0.3
mpl.rcParams['figure.subplot.left'] = 0.14
mpl.rcParams['figure.subplot.right'] = 0.95
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

def cg(X_train, y_train):
    adv_radius = get_radius(X_train, y_train, p=np.inf, option='randn_zero')
    estimated_params, info = lin_advregr(X_train, y_train, adv_radius=adv_radius, max_iter=100, p=np.inf, method='w-cg', utol=0)
    return estimated_params

def cholesky(X_train, y_train):
    adv_radius = get_radius(X_train, y_train, p=np.inf, option='randn_zero')
    estimated_params, info = lin_advregr(X_train, y_train, adv_radius=adv_radius, max_iter=100, p=np.inf, method='w-ridge', utol=0)
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
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    # Add argument for plotting
    parser.add_argument('--setting', choices=['spiked_covariance', 'sparse_gaussian', 'gaussian_overp',
                                            'comparing_advtrain_linf_methods'], default='comparing_advtrain_linf_methods')
    parser.add_argument('--plot_type', choices=['R2', 'time'], default='time')
    parser.add_argument('--dont_plot', action='store_true', help='Enable plotting')
    parser.add_argument('--dont_show', action='store_true', help='dont show plot, but maybe save it')
    parser.add_argument('--load_data', action='store_true', help='Enable data loading')
    parser.add_argument('--n_reps', type=int, default=5)
    parser.add_argument('--n_points', type=int, default=10)

    args = parser.parse_args()


    if args.setting == 'spiked_covariance':
        xlabel = r'\# components / \# features'
        all_methods = [advtrain_linf, lasso_cv]
        configs = np.linspace(1 / args.n_points, 1, args.n_points)
        def dset(alpha):
            n_train = 500
            n_params = 50
            n_latent = int(np.ceil(alpha * n_params))
            return latent_features(n_train, n_test, n_params, seed=rep, n_latent=n_latent, noise_std=0.1)
    elif args.setting == 'sparse_gaussian':
        xlabel = r'density'
        all_methods = [advtrain_linf, lasso_cv]
        configs = np.linspace(1 / args.n_points, 1, args.n_points)
        def dset(alpha):
            n_train = 500
            n_params = 50
            non_zeros = int(np.ceil(alpha * n_params))
            return sparse_gaussian(n_train, n_test, n_params, seed=rep, non_zeros=non_zeros, noise_std=0.1)
    elif args.setting == 'gaussian_overp':
        xlabel = r'\# features / \# samples'
        all_methods = [advtrain_linf, lasso_cv]
        configs = np.linspace(1 / args.n_points, 2, args.n_points)
        def dset(alpha):
            n_train = 500
            n_params = 2 * int(np.ceil(alpha * n_train))
            return gaussian(n_train, n_test, n_params, seed=rep, noise_std=0.1)
    elif args.setting == 'gaussian_sparse_overp':
        xlabel = r'\# features / \# samples'
        all_methods = [advtrain_linf, lasso_cv]
        configs = np.linspace(1 / args.n_points, 2, args.n_points)
        def dset(alpha):
            n_train = 500
            n_params = 2 * int(np.ceil(alpha * n_train))
            return sparse_gaussian(n_train, n_test, n_params, seed=rep, non_zeros=10,  noise_std=0.1)
    elif args.setting == 'spiked_covariance_overp':
        xlabel = r'\# features / \# samples'
        all_methods = [advtrain_linf, lasso_cv]
        configs = np.linspace(1 / args.n_points, 2, args.n_points)
        def dset(alpha):
            n_train = 500
            n_params = 2 * int(np.ceil(alpha * n_train))
            return latent_features(n_train, n_test, n_params, seed=rep, n_latent=1, noise_std=0.1)
    elif args.setting == 'comparing_advtrain_linf_methods':
        xlabel = r'\# samples'
        all_methods = [cholesky, cg]
        configs = np.arange(1, args.n_points) * 1000
        def dset(alpha):
            n_train = int(alpha)
            n_params = int(0.1 * n_train)
            print(n_train, n_params)
            return gaussian(n_train, n_test, n_params, seed=rep, noise_std=0.1)

    if args.load_data:
        print('loading data...')
        df = pd.read_csv(f'data/{args.setting}.csv')
    else :
        n_test = 500
        all_results = {'methods':  [''] * args.n_reps * len(configs) * len(all_methods),
                       'R2': np.zeros(args.n_reps * len(configs) * len(all_methods)),
                       'alpha': np.zeros(args.n_reps * len(configs) * len(all_methods)),
                       'time': np.zeros(args.n_reps * len(configs) * len(all_methods))}
        i = 0
        for alpha in configs:
            for rep in range(args.n_reps):
                X_train, X_test, y_train, y_test = dset(alpha)
                n_train, n_params = X_train.shape
                # Test dimension
                for method in all_methods:
                    start_time = time.time()
                    params = method(X_train, y_train)
                    exec_time = time.time() - start_time
                    y_pred = X_test @ params
                    all_results['R2'][i] = r2_score(y_test, y_pred)
                    all_results['time'][i] = exec_time
                    all_results['alpha'][i] = alpha
                    all_results['methods'][i] = method.__name__
                    i += 1
        df = pd.DataFrame(all_results)
        df.to_csv(f'data/{args.setting}.csv')

    if not args.dont_plot:
        fig, ax = plt.subplots()
        c = ['red', 'blue']
        for i, m in enumerate(all_methods):
            plot_errorbar(df[df['methods'] == m.__name__], 'alpha', args.plot_type, ax, m.__name__, color=c[i])
        plt.ylabel(args.plot_type)
        plt.xlabel(xlabel)
        if args.plot_type == 'R2':
            plt.ylim((0, 1))
        plt.legend()
        plt.savefig('imgs/' + args.setting + '.pdf')
        if not args.dont_show:
            plt.show()



