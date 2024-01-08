import numpy as np
from linadvtrain.datasets import load_magic
import sklearn.model_selection
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso
from ucimlrepo import fetch_ucirepo, list_available_datasets

def nerror(X_test, y_test, param):
    y_pred = X_test @ param
    return np.sqrt((y_test - y_pred) ** 2 / np.mean(y_test ** 2))

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
    F.loc['Sex'] = (F['Sex'] == 'M').values.astype(float)
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
    for dset in [heart_failure,]:
        df[dset.__name__] = pd.DataFrame({'dset': [], 'method': [], 'nerror': []})
        for method in [lasso_cv, advtrain]:
            X_train, X_test, y_train, y_test = dset()
            param = method(X_train, y_train)
            n_test = len(y_test)
            df1 = pd.DataFrame({'dset': n_test * [dset.__name__],
                                'method': n_test * [method.__name__],
                                'nerror': nerror(X_test, y_test, param)})

            df[dset.__name__] = pd.concat([df[dset.__name__], df1])
        df[dset.__name__].to_csv(f'fig1(a){dset.__name__}.csv')


    if load_data:
        df = {}
        for dset in [magic, diabetes, wine, abalone, heart_failure]:
            df[dset.__name__] = pd.read_csv(f'fig1(a){dset.__name__}.csv')

    df_final = pd.concat(df.values())

    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('./mystyle.mpl')
    sns.color_palette("hls", 8)
    sns.pointplot(data=df_final, x='dset', y='nerror', hue='method', dodge=.4, linestyles="none", errorbar=('pi', 0.5),
                  markers="o", markersizes=20, markeredgewidths=3,  capsize=0.05, n_boot=500, seed=10)
    plt.xlabel('')
    plt.ylabel('RMSE')
    plt.savefig('fig1(a).pdf')
    plt.show()

