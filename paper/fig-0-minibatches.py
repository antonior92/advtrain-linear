import numpy as np
from linadvtrain.datasets import load_magic
import sklearn.model_selection
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso
from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib
from linadvtrain.classification import lin_advclasif, CostFunction
import numpy as np
import linadvtrain.cvxpy_impl as cvxpy_impl
import time

# Additional style
mpl.rcParams['figure.figsize'] = 7, 3

mpl.rcParams['figure.subplot.left'] = 0.19
mpl.rcParams['figure.subplot.bottom'] = 0.25
mpl.rcParams['figure.subplot.right'] = 0.99
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 22
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.1
mpl.rcParams['xtick.major.pad'] = 7



X, y = datasets.load_breast_cancer(return_X_y=True)

X -= np.mean(X, axis=0)
X /= X.max(axis=0)  # Normalize each feature to be in [-1, 1], so adversarial training is fair
y = np.asarray(y, dtype=np.float64)


if __name__ == '__main__':
    adv_radius = 0.1
    n_train, n_params = X.shape
    # Compare with cvxpy
    start_time = time.time()
    mdl = cvxpy_impl.AdversarialClassification(X, y, p=np.inf)
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    exec_time = time.time() - start_time
    print(exec_time)


    # Test dimension
    n_iter = 50
    minibatch_list = [1, 10, 20, 50]
    dist_gd = np.empty([len(minibatch_list), n_iter])
    dist_gd[:] = np.nan
    for ll, mm in enumerate(minibatch_list):
        def callback(i, w, update_size):
            dist_gd[ll, i] = np.linalg.norm(w[:-1] - params_cvxpy)
        params, info = lin_advclasif(X, y, adv_radius=adv_radius, batch_size=mm,
                                     method='saga', callback=callback, verbose=True,
                                     p=np.inf, max_iter=n_iter, lr=3000)

    import matplotlib.pyplot as plt

    plt.plot(dist_gd.T, label=minibatch_list)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('# iter')
    plt.ylabel(r'$||\beta^{(i)} - \beta_*||$')
    plt.savefig('imgs/fig5.pdf')
    plt.show()

