import pytest
import sklearn.datasets
from linadvtrain import lin_advregr
from numpy import allclose
import linadvtrain.cvxpy_impl as cvxpy_impl
from linadvtrain.regression import lin_advregr, get_radius
import sklearn.model_selection
from datasets import magic
import numpy as np
import time
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

# Basic style
plt.style.use(['mystyle.mpl'])

# Additional style
mpl.rcParams['figure.figsize'] = 7, 3
mpl.rcParams['figure.subplot.bottom'] = 0.23
mpl.rcParams['figure.subplot.right'] = 0.99
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 22
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.1
mpl.rcParams['xtick.major.pad'] = 7

def get_diabetes():
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    # Standardize data (easier to set the l1_ratio parameter)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X, y

def normalize(X_train, X_test, y_train, y_test):
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    # Add argument for plotting
    parser.add_argument('--dont_plot', action='store_true', help='Enable plotting')
    parser.add_argument('--load_data', action='store_true', help='Enable data loading')
    parser.add_argument('--n_params', type=int, nargs='+', default=[30, 100, 300, 1000, 3000, 10000, 30000, 100000],
                        help='List of parameters to use (choose from [50, 100, 200])')
    parser.add_argument('--max_cvxpy', type=int, default=150,
                        )
    args = parser.parse_args()


    print('load magic..')
    X_train_, X_test_, y_train, y_test = magic()

    if args.load_data:
        df = pd.read_csv('data/fig1(b).csv')
    else:
        df = pd.DataFrame({'method': [], 'n_params': [], 'time': []})
        for n_params in args.n_params:
            X_train, X_test = X_train_[:, :n_params], X_test_[:, :n_params]
            print(f'running for {X_train.shape[1]}')
            n_train, n_params = X_train.shape
            # Test dimension
            adv_radius = get_radius(X_train, y_train, p=np.inf, option='randn_zero')

            start_time = time.time()
            print(f'ours')
            params, info = lin_advregr(X_train, y_train, adv_radius=adv_radius, verbose=False, p=np.inf, method='w-ridge')
            exec_time = time.time() - start_time
            df = df.append({
                'method': 'irr',
                'n_params': X_train.shape[1],
                'time': exec_time}, ignore_index=True)

            # Compare with cvxpy
            if n_params < args.max_cvxpy:
                start_time = time.time()
                print(f'cvxpy')
                mdl = cvxpy_impl.AdversarialRegression(X_train, y_train, p=np.inf)
                params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
                exec_time = time.time() - start_time
                df = df.append({
                    'method': 'cvxpy',
                    'n_params': X_train.shape[1],
                    'time': exec_time}, ignore_index=True)


                dist_sols = np.linalg.norm(params_cvxpy - params)
                print(f'distance = {dist_sols}')
            print(f'---')
            df.to_csv('data/fig1(b).csv', index=False)

    if not args.dont_plot:
        plt.figure()
        sns.color_palette("hls", 8)
        sns.pointplot(data=df, x='n_params', y='time', hue='method', errorbar=None, native_scale=True)
        plt.xscale('log')
        plt.xlabel('\# parameters')
        plt.ylabel('time (s)')
        plt.yscale('log')
        plt.legend(title='', labels=['ours', 'cvxpy'])
        plt.savefig('imgs/fig1(b).pdf')
        plt.show()
