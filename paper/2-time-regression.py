
import sklearn.datasets
import linadvtrain.cvxpy_impl as cvxpy_impl
from linadvtrain.regression import lin_advregr, get_radius
import sklearn.model_selection
from datasets import magic
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso

# Basic style
plt.style.use(['mystyle.mpl'])

# Additional style
mpl.rcParams['figure.figsize'] = 7, 3
mpl.rcParams['figure.subplot.bottom'] = 0.23
mpl.rcParams['figure.subplot.right'] = 0.99
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 22
mpl.rcParams['legend.fontsize'] = 17
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.5
mpl.rcParams['legend.labelspacing'] = 0.2
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
    parser.add_argument('--dont_show', action='store_true', help='Enable plotting')
    parser.add_argument('--load_data', action='store_true', help='Enable data loading')
    parser.add_argument('--n_params', type=int, nargs='+',  default=[30, 100, 300, 1000, 3000, 10000, 30000, 100000],
                        help='List of parameters to use')
    parser.add_argument('--n_rep', type=int, default=2,
                        help="number of repetitions")
    parser.add_argument('--max_cvxpy', type=int, default=150,
                        )
    args = parser.parse_args()



    print('load magic..')
    X_train_, X_test_, y_train, y_test = magic()

    if args.load_data:
        df = pd.read_csv('data/time-regression.csv')
    else:
        df = pd.DataFrame({'method': [], 'n_params': [], 'time': []})
        for i in range(args.n_rep):
            rng = np.random.RandomState(i)
            for n_params in args.n_params:
                indices = rng.choice(X_train_.shape[1], size=n_params)
                X_train = X_train_[:, indices]
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

                # Compare with cross validated lasso
                start_time = time.time()
                regr = ElasticNetCV(cv=10, random_state=0, l1_ratio=1)
                regr.fit(X_train, y_train)
                exec_time = time.time() - start_time
                df = df.append({
                    'method': 'cvlasso',
                    'n_params': X_train.shape[1],
                    'time': exec_time}, ignore_index=True)

                print(f'---')
                df.to_csv('data/time-regression.csv', index=False)

    if not args.dont_plot:
        import seaborn as sns
        plt.figure()
        with sns.color_palette("Set2"):
            g =  sns.pointplot(data=df, x='n_params', y='time', hue='method', errorbar='ci', native_scale=True,
                          hue_order=['cvxpy', 'cvlasso', 'irr'])
            plt.legend(title='')
            for t, l in zip(g.legend_.texts, ['AdvTrain-CVXPY', 'Lasso CV', 'AdvTrain-Ours']):
                t.set_text(l)
        plt.xscale('log')
        plt.xlabel('\# parameters')
        plt.ylabel('time (s)')
        plt.yscale('log')
        plt.savefig('imgs/time-regression.pdf')
        if not args.dont_show:
            plt.show()
