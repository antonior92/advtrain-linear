import numpy as np
from linadvtrain.datasets import load_magic
import sklearn.model_selection
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso
from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt


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
mpl.rcParams['legend.handletextpad'] = 0.01
mpl.rcParams['xtick.major.pad'] = 7

def compute_R2(y_pred, y_test):
    y_mean = np.mean(y_test)
    return 1 - np.mean((y_test - y_pred) ** 2) / np.mean((y_test - y_mean) ** 2)

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

# DATASET
def magic():
    X, y = load_magic(input_folder='../WEBSITE/DATA')
    # Train-test split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50, random_state=0)
    X_train = X_train[:, :1000]
    X_test = X_test[:, :1000]
    return normalize(X_train, X_test, y_train, y_test)

def diabetes():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50, random_state=0)
    return normalize(X_train, X_test, y_train, y_test)

def wine():
    dset = fetch_ucirepo(name="Wine Quality")
    X = dset.data.features.values
    y = dset.data.targets.values.flatten()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    return normalize(X_train, X_test, y_train, y_test)

def abalone():
    dset = fetch_ucirepo(name="Abalone")
    F = dset.data.features
    F = F.assign(Sex=(F['Sex'] == 'M').values.astype(float))
    X = F.values
    y = dset.data.targets.values.flatten()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    return normalize(X_train, X_test, y_train, y_test)

def heart_failure():
    dset = fetch_ucirepo(name="Heart failure clinical records")
    F = dset.data.features
    X = F.values
    y = dset.data.targets.values.flatten()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    return normalize(X_train, X_test, y_train, y_test)

# METHODS
def lasso_cv(X_train, y_train):
    regr = ElasticNetCV(cv=10, random_state=0, l1_ratio=1)
    regr.fit(X_train, y_train)
    return regr.coef_

def lasso_aic(X_train, y_train):
    regr = LassoLarsIC(criterion='aic')
    regr.fit(X_train, y_train)
    return regr.coef_

def lasso_bic(X_train, y_train):
    regr = LassoLarsIC(criterion='bic')
    regr.fit(X_train, y_train)
    return regr.coef_

def advtrain(X_train, y_train):
    estimated_params, info = lin_advregr(X_train, y_train, p=np.inf)
    return estimated_params

def lasso(X_train, y_train):
    regr = Lasso()
    regr.fit(X_train, y_train)
    return regr.coef_


if __name__ == '__main__':
    from linadvtrain.regression import lin_advregr
    import pandas as pd

    load_data=True

    # Compute performance
    df = {}
    for dset in []:
        df[dset.__name__] = pd.DataFrame({'dset': [], 'method': [], 'y_pred': []})
        for method in [lasso_cv, advtrain]:
            X_train, X_test, y_train, y_test = dset()
            param = method(X_train, y_train)
            n_test = len(y_test)
            df1 = pd.DataFrame({'dset': n_test * [dset.__name__],
                                'method': n_test * [method.__name__],
                                'y_pred': X_test @ param,
                                'y_test': y_test})

            df[dset.__name__] = pd.concat([df[dset.__name__], df1])
        df[dset.__name__].to_csv(f'fig1(a){dset.__name__}.csv')

    if load_data:
        df = {}
        for dset in [diabetes, wine, abalone, heart_failure]:
            df[dset.__name__] = pd.read_csv(f'fig1(a){dset.__name__}.csv')

    df_final = pd.DataFrame({'dset': [], 'method': [], 'R2': []})
    for dset in [heart_failure, diabetes, wine, abalone]:
        for method in [lasso_cv, advtrain]:
            df_aux = df[dset.__name__].loc[df[dset.__name__]['method'] == method.__name__]

            value_r2 = compute_R2(df_aux['y_pred'], df_aux['y_test'])

            n_boot = 500
            value_r2_bootstrap = np.zeros(n_boot)
            for i in range(n_boot):
                indexes = np.random.choice(range(len(df_aux['y_pred'])), len(df_aux['y_pred']))
                df_permuted = df_aux.iloc[indexes]
                value_r2_bootstrap[i] = compute_R2(df_permuted['y_pred'], df_permuted['y_test'])

            df_final = df_final.append({
                'dset': dset.__name__,
                'method': method.__name__,
                'R2': value_r2}, ignore_index=True)

            df_final = df_final.append({
                'dset': dset.__name__,
                'method': method.__name__,
                'R2': np.quantile(value_r2_bootstrap, 0.025)}, ignore_index=True)

            df_final = df_final.append({
                'dset': dset.__name__,
                'method': method.__name__,
                'R2': np.quantile(value_r2_bootstrap, 0.975)}, ignore_index=True)


    plt.figure()
    sns.color_palette("hls", 8)
    sns.pointplot(data=df_final, x='dset', y='R2', hue='method', dodge=.4, estimator='median', linestyles="none",
                  errorbar=('pi', 100), capsize=0.05)
    plt.xlabel('')
    plt.ylabel('$$R^2$$')
    plt.ylim((0, 1))
    plt.legend(title='')
    plt.savefig('fig1(a).pdf')
    plt.show()

