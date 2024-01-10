import pytest
import sklearn.datasets
from linadvtrain import lin_advregr
from numpy import allclose
import linadvtrain.cvxpy_impl as cvxpy_impl
from linadvtrain.regression import lin_advregr, get_radius
import sklearn.model_selection
from linadvtrain.datasets import load_magic
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

    load_data = True


    X, y = load_magic(input_folder='../WEBSITE/DATA')

    # Train-test split
    X_train_, X_test_, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50, random_state=0)
    X_train_, X_test_, y_train, y_test = normalize(X_train_, X_test_, y_train, y_test)

    if load_data:
        df = pd.read_csv('data/fig1(b).csv')
    else:
        df = pd.DataFrame({'method': [], 'n_params': [], 'time': []})
    for n_params in []:
        X_train, X_test = X_train_[:, :n_params], X_test_[:, :n_params]
        n_train, n_params = X_train.shape
        # Test dimension
        adv_radius = get_radius(X_train, y_train, p=np.inf, option='randn_zero')

        start_time = time.time()
        params, info = lin_advregr(X_train, y_train, adv_radius=adv_radius, verbose=False, p=np.inf, method='w-ridge')
        exec_time = time.time() - start_time
        df = df.append({
            'method': 'irr',
            'n_params': n_params,
            'time': exec_time}, ignore_index=True)

        # Compare with cvxpy
        start_time = time.time()
        mdl = cvxpy_impl.AdversarialRegression(X_train, y_train, p=np.inf)
        params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
        exec_time = time.time() - start_time
        df = df.append({
            'method': 'cvxpy',
            'n_params': n_params,
            'time': exec_time}, ignore_index=True)
        df.to_csv('fig1(b).csv', index=False)


    plt.figure()
    sns.color_palette("hls", 8)
    sns.pointplot(data=df, x='n_params', y='time', hue='method', errorbar=None, native_scale=True)
    plt.xscale('log')
    plt.xlabel('\# parameters')
    plt.xlabel('time (s)')
    plt.yscale('log')
    plt.legend(title='', labels=['ours', 'cvxpy'])
    plt.savefig('fig1(b).pdf')
    plt.show()
