
import sklearn.datasets
import linadvtrain.cvxpy_impl as cvxpy_impl
from linadvtrain.regression import lin_advregr, get_radius
import sklearn.model_selection
from datasets import magic
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from datasets import diabetes
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


if __name__ == "__main__":
    n_iter = 100
    min_fs = np.inf
    configs = [{'method': 'w-ridge'},
               {'method': 'w-cg'}]

    labels = [r'IRRR', 'ICG']
    X_train, X_test, y_train, y_test = diabetes()
    adv_radius = get_radius(X_train, y_train, p=np.inf, option='randn_zero')

    mdl = cvxpy_impl.AdversarialRegression(X_train, y_train, p=np.inf)
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    min_fs = np.mean((np.abs(X_train @ params_cvxpy - y_train) + adv_radius * np.linalg.norm(params_cvxpy, ord=1))**2)

    fs = np.empty([2, n_iter + 1])
    fs[:] = np.nan
    for ll, config in enumerate(configs):
        def cb(i, params, loss):
            fs[ll, i] = loss

        params, info = lin_advregr(X_train, y_train, adv_radius=adv_radius, verbose=True, p=np.inf,
                                   callback=cb, **config)

    colors = ['b', 'g', 'r', 'c', 'k']
    linestyle = [':', ':', ':', ':', '-']
    plt.figure()
    for i in range(fs.shape[0]):
        plt.plot(range(n_iter + 1), fs[i, :] - min_fs, label=labels[i], color=colors[i], ls=linestyle[i])
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel('\# iter')
    plt.ylim([1e-8, (fs[~np.isnan(fs)] - min_fs).max()])
    plt.ylabel(r'$R^{(i)} - R_*$')
    plt.show()




