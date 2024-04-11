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


def get_mnist():
    image_size = 28  # width and length
    no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = "../dsets/"
    train_data = np.loadtxt(data_path + "mnist_train.csv",
                            delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv",
                           delimiter=",")
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    return train_imgs, test_imgs, (train_labels == 0).flatten(), (test_labels == 0).flatten()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score


    X_train, X_test, y_train, y_test = get_mnist()
    n_iter = 10
    max_config = 3

    def power_method_covmatr(X, num_iterations: int = 10):
        # Here A = 1/n X.T X
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        np.random.seed(1)
        v = np.random.rand(X.shape[0])
        n = X.shape[0]

        for _ in range(num_iterations):
            v = 1/n * np.dot(X, np.dot(X.T, v))
            s = np.max(v)
            v = v / s
        return s

    s_max = power_method_covmatr(X_train)
    lr=2/s_max


    accuracy_gd = np.empty([max_config, n_iter])
    def callback(i, w, update_size):
        roc_auc = roc_auc_score(y_test, X_test @ w[:-1])
        accuracy_gd[0, i] = roc_auc
    params, info = lin_advclasif(X_train, y_train, adv_radius=0.1, method='gd', verbose=True,
                                 p=np.inf, max_iter=n_iter, callback=callback)
    def callback(i, w, update_size):
        roc_auc = roc_auc_score(y_test, X_test @ w[:-1])
        accuracy_gd[1, i] = roc_auc
    params, info = lin_advclasif(X_train, y_train, adv_radius=0.1, method='sgd', verbose=True, lr=lr,
                                 p=np.inf, max_iter=n_iter, callback=callback)
    def callback(i, w, update_size):
        roc_auc = roc_auc_score(y_test, X_test @ w[:-1])
        accuracy_gd[2, i] = roc_auc
    params, info = lin_advclasif(X_train, y_train, adv_radius=0.1, method='saga', verbose=True,lr=lr,
                                 p=np.inf, max_iter=n_iter, callback=callback)

    plt.plot(accuracy_gd.T, label=['GD', 'SGD', 'SAGA'])
    #plt.yscale('log')
    plt.legend()
    plt.xlabel('# iter')
    plt.ylabel(r'Performance')
    plt.savefig('imgs/fig2(c).pdf')
    plt.show()
