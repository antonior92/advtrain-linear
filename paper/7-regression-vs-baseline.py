
import sklearn.datasets
import linadvtrain.cvxpy_impl as cvxpy_impl
from linadvtrain.regression import lin_advregr, get_radius
import sklearn.model_selection
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from datasets import *
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, Lasso
# Basic style
plt.style.use(['mystyle.mpl'])

# Additional style
mpl.rcParams['figure.figsize'] = 7, 3
mpl.rcParams['figure.subplot.bottom'] = 0.23
mpl.rcParams['figure.subplot.right'] = 0.99
mpl.rcParams['figure.subplot.left'] = 0.14
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 22
mpl.rcParams['legend.fontsize'] = 17
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.5
mpl.rcParams['legend.labelspacing'] = 0.2
mpl.rcParams['xtick.major.pad'] = 7


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    # Add argument for plotting
    parser.add_argument('--dset', choices=['abalone', 'wine', 'magic', 'diabetes'], default='abalone')
    parser.add_argument('--dont_plot', action='store_true', help='Enable plotting')
    parser.add_argument('--dont_show', action='store_true', help='dont show plot, but maybe save it')
    parser.add_argument('--load_data', action='store_true', help='Enable data loading')
    parser.add_argument('--n_iter', type=int, default=100)
    args = parser.parse_args()


    n_iter = 100
    min_fs = np.inf
    configs = [{'method': 'w-ridge'},
               {'method': 'w-cg'}]

    labels = [r'Cholesky', 'CG']
    dset = eval(args.dset)
    X_train, X_test, y_train, y_test = dset()
    adv_radius = get_radius(X_train, y_train, 'randn_zero', np.Inf)

    mdl = cvxpy_impl.AdversarialRegression(X_train, y_train, p=np.inf)
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    min_fs = np.mean((np.abs(X_train @ params_cvxpy - y_train) + adv_radius * np.linalg.norm(params_cvxpy, ord=1)) ** 2)


    fs = np.empty([2, n_iter + 1])
    fs[:] = np.nan
    for ll, config in enumerate(configs):
        def cb(i, params, loss):
            fs[ll, i] = loss

        params, info = lin_advregr(X_train, y_train, adv_radius=adv_radius, verbose=True, p=np.inf,
                                   callback=cb, max_iter=args.n_iter, **config)


    import torch

    num_steps = 10000
    lr = 0.1
    X = torch.tensor(X_train, dtype=torch.float)
    y = torch.tensor(y_train, dtype=torch.float)

    param = torch.randn(X.size(1), requires_grad=True)

    loss_gd = np.zeros(num_steps)
    for step in range(num_steps):
        param.requires_grad_(True)
        # Update loss
        param_norm = torch.linalg.norm(param, ord=1)
        abs_error = torch.abs(X @ param - y)
        loss = (1 / X.size(0)) * torch.sum((abs_error + adv_radius * param_norm) ** 2)

        loss.backward()
        with torch.no_grad():
            param -= lr / np.sqrt(step+1) * param.grad
            loss_gd[step] = loss.detach().numpy()
        param.grad.zero_()



    colors = ['b', 'g', 'r', 'c', 'k']
    linestyle = ['-', '-', ':', ':', '-']
    plt.figure()
    for i in range(fs.shape[0]):
        plt.plot(range(fs.shape[1]), fs[i, :] - min_fs, label=labels[i], color=colors[i], ls=linestyle[i])

    plt.plot(np.arange(num_steps)/ X_train.shape[1], loss_gd  - min_fs, label='GD', color='red', ls=':')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel('\# iter')
    plt.xlim([-1, np.minimum(args.n_iter, fs.shape[1]) + 1])
    #plt.ylim([1e-8, (fs[~np.isnan(fs)] - min_fs).max()])
    plt.ylabel('sub-optimality')

    plt.show()




