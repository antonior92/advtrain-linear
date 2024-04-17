import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso, Ridge, RidgeCV
from datasets import *
import matplotlib as mpl
import seaborn as sns
import time
import matplotlib.pyplot as plt
from linadvtrain.regression import lin_advregr
from  sklearn import ensemble, neural_network

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error

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

# DATASET REGRESSION

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




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    all_methods = [lasso_cv, advtrain_linf, lasso_bic, lasso_aic, lasso, ridge, ridgecv, advtrain_l2, gboost, mlp]
    datasets = [diabetes, wine, abalone, heart_failure]

    all_results= []
    for dset in datasets:
        X_train, X_test, y_train, y_test = dset()

        for method in all_methods:
            n_test = len(y_test)
            start_time = time.time()
            if method in [gboost, mlp]:
                reg = method(X_train, y_train)
                y_pred = reg.predict(X_test)
            else:
                param = method(X_train, y_train)
                y_pred = X_test @ param
            exec_time = time.time() - start_time
            #sns.scatterplot(x=y_test, y=y_pred).set_title(method.__name__)
            #plt.show()
            ms = [dset.__name__, method.__name__]
            ms += [m(y_test, y_pred) for m in [root_mean_squared_error, r2_score]]
            ms += bootstrap(y_test, y_pred, root_mean_squared_error, [0.25, 0.75])
            ms += bootstrap(y_test, y_pred, r2_score, [0.25, 0.75])
            ms += [exec_time]
            all_results.append(ms)

    df = pd.DataFrame(all_results, columns=['dset', 'method', 'RMSE', 'R2',  'RMSEq1', 'RMSEq3', 'R2q1', 'R2q3', 'exec_time'])

    df.to_csv(f'data/performace_regression.csv')

    print(df)

    # Plot figure
    fig, ax = plt.subplots()
    df_lasso_cv = df[df['method'] == 'lasso_cv']
    df_advtrain_linf = df[df['method'] == 'advtrain_linf']
    width = 0.35
    ind = np.arange(len(datasets))

    metric_show = 'R2'
    ddf = df_lasso_cv
    y_err = [ddf[metric_show] - ddf[metric_show + 'q1'], ddf[metric_show + 'q3']- ddf[metric_show]]
    rects1 = ax.bar(ind - width / 2,  ddf[metric_show], width, yerr=y_err, label='lasso_cv')

    ddf = df_advtrain_linf
    y_err = [ddf[metric_show] - ddf[metric_show + 'q1'], ddf[metric_show + 'q3'] - ddf[metric_show]]
    rects2= ax.bar(ind + width / 2,  ddf[metric_show], width, yerr=y_err, label='advtrain_linf')

    plt.xticks([0, 1, 2, 3], [d.__name__ for d in datasets])
    plt.ylabel('$$R^2$$')
    plt.ylim((0, 1))
    plt.legend(['Lasso CV', 'Adv Train'], title='')
    plt.savefig('imgs/performace_regression.pdf')
    plt.show()
