import numpy as np
from linadvtrain.datasets import load_magic
import sklearn.model_selection
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso
from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets
from linadvtrain.classification import lin_advclasif, CostFunction
import numpy as np
import linadvtrain.cvxpy_impl as cvxpy_impl
import time


# Fig 2(a)
comparing_lr_iter = 1000
comparing_lr = [{'method': 'gd', 'backtrack': False, 'lr': '2/L'},
                {'method': 'gd', 'backtrack': False, 'lr': '10/L'},
                {'method': 'gd', 'backtrack': False, 'lr': '40/L'},
                {'method': 'gd', 'backtrack': False, 'lr': '160/L'},
                {'method': 'gd', 'backtrack': True}]
labels_lr = [r'lr = 2/$\lambda_{\mathrm{max}}$', 'lr = 10/$\lambda_{\mathrm{max}}$',
          'lr = 40/$\lambda_{\mathrm{max}}$', 'lr = 160/$\lambda_{\mathrm{max}}$', 'Backtrack LS']

# Fig 2(b)
comparing_acceleration_iter = 100
comparing_acceleration = [{'method': 'gd', 'backtrack': True},
                          {'method': 'agd', 'backtrack': True}]
labels_acceleration = ['GD', 'AGD']


# Fig 2(c)
comparing_stochastic_iter = 100
comparing_stochastic = [{'method': 'gd', 'backtrack': True},
                        {'method': 'sgd', 'lr': '200/L'},
                        {'method': 'saga'}]
labels_stochastic = ['GD', 'SGD', 'SAGA']

# Basic style
plt.style.use(['mystyle.mpl'])

# Additional style
mpl.rcParams['figure.figsize'] = 7, 3
mpl.rcParams['figure.subplot.left'] = 0.17
mpl.rcParams['figure.subplot.bottom'] = 0.23
mpl.rcParams['figure.subplot.right'] = 0.99
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 22
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.1
mpl.rcParams['xtick.major.pad'] = 7
plt.rcParams['image.cmap'] = 'gray'


X, y = datasets.load_breast_cancer(return_X_y=True)

X -= np.mean(X, axis=0)
X /= X.max(axis=0)  # Normalize each feature to be in [-1, 1], so adversarial training is fair
y = np.asarray(y, dtype=np.float64)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    compare_with_cvxpy = True
    adv_radius = 0.1
    n_train, n_params = X.shape
    configs = comparing_stochastic
    n_iter = comparing_stochastic_iter
    labels = labels_stochastic

    if compare_with_cvxpy:
        # Compare with cvxpy
        start_time = time.time()
        mdl = cvxpy_impl.AdversarialClassification(X, y, p=np.inf)
        params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
        exec_time = time.time() - start_time
        print(exec_time)
        min_fs = np.inf
    else:
        min_fs = np.inf


    fs = np.empty([len(configs), n_iter])
    for ll, config in enumerate(configs):
        start_time = time.time()
        params, info = lin_advclasif(X, y, adv_radius=adv_radius,
                                     verbose=True, p=np.inf, max_iter=n_iter, **config)
        exec_time = time.time() - start_time
        fs[ll, :] = info['costs']
        print(ll, exec_time)
        assert params.shape == (n_params,)

    colors = ['b', 'g', 'r', 'c', 'k']
    linestyle = [':', ':', ':', ':', '-']
    plt.figure()
    min_fs = min(fs[~np.isnan(fs)].min(), min_fs)
    for i in range(fs.shape[0]):
        plt.plot(range(n_iter), fs[i, :] - min_fs, label=labels[i], color=colors[i], ls=linestyle[i])
    plt.yscale('log')
    plt.legend()
    plt.xlabel('\# iter')
    plt.ylim([0.2e-7, max(fs[i, :] - min_fs)])
    plt.ylabel(r'$R^{(i)} - R_*$')
    plt.savefig('imgs/fig2(a).pdf')
    plt.show()


