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


if __name__ == "__main__":
    import pandas as pd

    rng = np.random.RandomState(1)

    if False:
        df = pd.read_csv('data/fig3.csv')
    else:
        df = pd.DataFrame({'method': [], 'n_params': [], 'time': []})
    for n_train in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                    2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]:
        for rep in range(3):
            n_params = int(0.1*n_train)
            X_train = rng.randn(n_train, n_params)
            beta = rng.randn(n_params)
            y_train = X_train @ beta + 0.1 * rng.randn(n_train)
            n_train, n_params = X_train.shape
            # Test dimension
            adv_radius = get_radius(X_train, y_train, p=np.inf, option='randn_zero')

            start_time = time.time()
            params, info = lin_advregr(X_train, y_train, adv_radius=adv_radius, max_iter=100, p=np.inf, method='w-ridge', utol=0)
            exec_time = time.time() - start_time
            df = df.append({
                'method': 'irr',
                'n_params': n_params,
                'time': exec_time}, ignore_index=True)

            # Compare with cvxpy
            start_time = time.time()
            paramscg, info = lin_advregr(X_train, y_train, adv_radius=adv_radius,  p=np.inf, max_iter=100, method='w-cg', utol=0)
            exec_time = time.time() - start_time
            df = df.append({
                'method': 'cg',
                'n_params': n_params,
                'time': exec_time}, ignore_index=True)


            #start_time = time.time()
            #mdl = cvxpy_impl.AdversarialRegression(X_train, y_train, p=np.inf)
            #params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
            #exec_time = time.time() - start_time
            #df = df.append({
            #    'method': 'cvxpy',
            #    'n_params': n_params,
            #    'time': exec_time}, ignore_index=True)


            df.to_csv('data/fig3.csv', index=False)

            #print(n_train, np.linalg.norm(paramscg- params_cvxpy))


    plt.figure()
    sns.pointplot(data=df, x='n_params', y='time', hue='method', errorbar=None, native_scale=True,
                  palette ='colorblind', estimator='median')
    plt.xlabel('\# parameters')
    plt.legend(title='')
    plt.savefig('imgs/fig3.pdf')
    plt.show()
