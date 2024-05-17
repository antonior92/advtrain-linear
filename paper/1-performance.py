import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso, Ridge, RidgeCV, LogisticRegression
from linadvtrain.classification import lin_advclasif
from datasets import *
import matplotlib as mpl
import seaborn as sns
import time
import matplotlib.pyplot as plt
from linadvtrain.regression import lin_advregr, get_radius
from  sklearn import ensemble, neural_network

from sklearn.metrics import (r2_score, root_mean_squared_error, mean_absolute_percentage_error, roc_auc_score,
                             average_precision_score)

# Basic style
plt.style.use(['mystyle.mpl'])


# Additional style
mpl.rcParams['figure.figsize'] = 7, 3
mpl.rcParams['figure.subplot.bottom'] = 0.25
mpl.rcParams['figure.subplot.right'] = 0.97
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 22
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --------------------------
# --- Regression Methods ---
# --------------------------
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

def advtrain_linf(X_train, y_train):
    estimated_params, info = lin_advregr(X_train, y_train, p=np.inf)
    return estimated_params

def advtrain_l2(X_train, y_train):
    estimated_params, info = lin_advregr(X_train, y_train, p=2)
    return estimated_params

def lasso(X_train, y_train):
    regr = Lasso()
    regr.fit(X_train, y_train)
    return regr.coef_

def ridge(X_train, y_train):
    regr = Ridge(alpha = 100)
    regr.fit(X_train, y_train)
    return regr.coef_

def ridgecv(X_train, y_train):
    regr = RidgeCV()
    regr.fit(X_train, y_train)
    return regr.coef_

def gboost(X_train, y_train):
    reg = ensemble.GradientBoostingRegressor()
    reg.fit(X_train, y_train)
    return reg

def mlp(X_train, y_train):
    reg = neural_network.MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    return reg

def bootstrap(y_test, y_pred, metric, quantiles, n_boot=500):
    value_r2_bootstrap = np.zeros(n_boot)
    for i in range(n_boot):
        indexes = np.random.choice(range(len(y_pred)), len(y_pred))
        value_r2_bootstrap[i] = metric(y_test[indexes], y_pred[indexes])
    return [np.quantile(value_r2_bootstrap, q) for q in quantiles]


# ------------------------------
# --- Classification Methods ---
# ------------------------------
def logistic(X_train, y_train):
    clf = LogisticRegression(random_state=0, fit_intercept=False).fit(X_train, y_train)
    return clf

def advclassif_linf(X_train, y_train):
    params, info = lin_advclasif(X_train, y_train, p=2, method='agd', max_iter=1000,
                                adv_radius=1e-20)
    return params


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', choices=['regr', 'classif'], default='regr')
    parser.add_argument('--dont_plot', action='store_true', help='Enable plotting')
    parser.add_argument('--dont_show', action='store_true', help='dont show plot, but maybe save it')
    parser.add_argument('--load_data', action='store_true', help='Enable data loading')
    args = parser.parse_args()



    if args.setting == 'regr':
        all_methods = [advtrain_linf, lasso_cv,  lasso, advtrain_l2, ridge, ridgecv, gboost, mlp]
        datasets = [diabetes, wine, abalone, heartf, polution, diamonds]
        tp = 'regression'
        metrics_names = ['RMSE', 'R2']
        metrics_of_interest = [root_mean_squared_error, r2_score]
        metric_show = 'R2'
        ylabel = '$$R^2$$'
        methods_to_show = ['advtrain_linf', 'lasso_cv']
        methods_name = ['Adv Train', 'Lasso CV']
    elif args.setting == 'classif':
        all_methods = [advclassif_linf, logistic]
        tp = 'classification'
        datasets = [breast_cancer, iris]
        metrics_names = ['AUROC', 'AUPRC']
        metrics_of_interest = [roc_auc_score, average_precision_score]
        methods_to_show = ['advclassif_linf', 'logistic']
        methods_name = ['Adv Train', 'Logistic']
        metric_show = 'AUROC'
        ylabel = metric_show

    columns_names = ['dset', 'method'] + metrics_names + \
                    [nn + q for nn in metrics_names for q in ['q1', 'q3']] +\
                    ['exec_time']
    all_results= []
    for dset in datasets:
        X_train, X_test, y_train, y_test = dset()

        for method in all_methods:
            n_test = len(y_test)
            start_time = time.time()
            if method in [gboost, mlp]:
                reg = method(X_train, y_train)
                y_pred = reg.predict(X_test)
            elif method in [logistic, ]:
                clf = method(X_train, y_train)
                y_pred = clf.predict_proba(X_test)[:, 1:]
            elif method in [advclassif_linf,]:
                params = method(X_train, y_train)
                y_pred = sigmoid(X_test @ params)
            else:
                print(X_train.shape, y_train.shape)
                params = method(X_train, y_train)
                y_pred = X_test @ params
            exec_time = time.time() - start_time
            #sns.scatterplot(x=y_test, y=y_pred).set_title(method.__name__)
            #plt.show()
            ms = [dset.__name__, method.__name__]
            ms += [m(y_test, y_pred) for m in metrics_of_interest]
            for m in metrics_of_interest:
                ms += bootstrap(y_test, y_pred, m, [0.25, 0.75])
            ms += [exec_time]
            all_results.append(ms)

    df = pd.DataFrame(all_results, columns=columns_names)

    df.to_csv(f'data/performace_{args.setting}.csv')

    print(df)

    for nn in metrics_names:
        print(nn)
        # Also print preprocessed version for the paper
        ddf = df[['dset', 'method', nn]].set_index(['dset', 'method']).iloc[:, 0]
        ddf = ddf.unstack('method')
        ddf.index.name = None
        print(ddf.to_latex(columns=[m.__name__ for m in all_methods], float_format="%.2f"))

    # Plot figure
    from matplotlib import ticker

    mpl.rcParams['xtick.major.pad'] = 7
    mpl.rcParams['xtick.minor.pad'] = 25
    mpl.rcParams['xtick.direction'] = 'in'
    fig, ax = plt.subplots()
    width = 0.35
    ind = np.arange(len(datasets))

    for i in range(2):
        ddf = df[df['method'] == methods_to_show[i]]
        y_err = [ddf[metric_show] - ddf[metric_show + 'q1'], ddf[metric_show + 'q3'] - ddf[metric_show]]
        ii = ind - width / 2 if i == 0 else ind + width / 2
        rects1 = ax.bar(ii,  ddf[metric_show], width, yerr=y_err, label=methods_name[i])

    plt.xticks(range(len(datasets)), [d.__name__ for d in datasets])
    plt.ylabel(ylabel)
    plt.ylim((0, 1))
    plt.legend( title='')

    ax = plt.gca()
    major_names = [d.__name__ for i, d in enumerate(datasets) if i % 2 == 0]
    minor_names = [d.__name__ for i, d in enumerate(datasets) if i % 2 == 1]
    major_loc = [i for i, d in enumerate(datasets) if i % 2 == 0]
    minor_loc = [i for i, d in enumerate(datasets) if i % 2 == 1]
    ax.xaxis.set_major_locator(ticker.FixedLocator(major_loc))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_loc))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(major_names))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(minor_names))
    ax.tick_params(axis='x', which='minor', length=-200)
    ax.tick_params(axis='x', which='both', color='lightgrey')
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.savefig(f'imgs/performace_{tp}.pdf')
    plt.show()
