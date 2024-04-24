import matplotlib as mpl
import matplotlib.pyplot as plt
from linadvtrain.classification import lin_advclasif, CostFunction
import numpy as np
import linadvtrain.cvxpy_impl as cvxpy_impl
import time
from datasets import *


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    # Add argument for plotting
    parser.add_argument('--dset', choices=['breast_cancer', 'mnist', 'magic_classif'], default='breast_cancer')
    parser.add_argument('--setting', choices=['compare_lr', 'acceleration', 'stochastic'], default='compare_lr')
    parser.add_argument('--dont_plot', action='store_true', help='Enable plotting')
    parser.add_argument('--dont_show', action='store_true', help='Enable plotting')
    parser.add_argument('--load_data', action='store_true', help='Enable data loading')
    args = parser.parse_args()

    adv_radius = 0.1


    dset = eval(args.dset)
    X, _x, y, _y = dset()

    compare_with_cvxpy = args.dset in ['breast_cancer', ]  # dont compare for other datasets since they are too big

    # Fig 2(a)
    if args.setting == 'compare_lr':
        n_iter = 1000
        configs = [{'method': 'gd', 'backtrack': False, 'lr': '2/L'},
                   {'method': 'gd', 'backtrack': False, 'lr': '10/L'},
                   {'method': 'gd', 'backtrack': False, 'lr': '40/L'},
                   {'method': 'gd', 'backtrack': False, 'lr': '160/L'},
                   {'method': 'gd', 'backtrack': True}]
        labels = [r'lr = 2/$\lambda_{\mathrm{max}}$', 'lr = 10/$\lambda_{\mathrm{max}}$',
                     'lr = 40/$\lambda_{\mathrm{max}}$', 'lr = 160/$\lambda_{\mathrm{max}}$', 'Backtrack LS']
    # Fig 2(b)
    elif args.setting == 'acceleration':
        n_iter = 100
        configs = [{'method': 'gd', 'backtrack': True},
                   {'method': 'agd', 'backtrack': True}]
        labels = ['GD', 'AGD']

    # Fig 2(c)
    elif args.setting == 'stochastic':
        n_iter = 100
        configs = [{'method': 'gd', 'backtrack': True},
                   {'method': 'sgd', 'lr': '200/L'},
                   {'method': 'saga'}]
        labels = ['GD', 'SGD', 'SAGA']


    if args.load_data:
        print('loading data...')
        fs = np.loadtxt(f'data/{args.setting}_{args.dset}.csv')
        min_fs = np.inf
    else:
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

        print('start')
        fs = np.empty([len(configs), n_iter + 1])
        for ll, config in enumerate(configs):
            print(f'config {ll}, {config}')
            start_time = time.time()
            params, info = lin_advclasif(X, y, adv_radius=adv_radius,
                                         verbose=True, p=np.inf, max_iter=n_iter, **config)
            print(info)
            exec_time = time.time() - start_time
            fs[ll, :] = info['costs']
            print(ll, exec_time)
        np.savetxt(f'data/{args.setting}_{args.dset}.csv', fs)

    if not args.dont_plot:
        colors = ['b', 'g', 'r', 'c', 'k']
        linestyle = [':', ':', ':', ':', '-']
        plt.figure()
        min_fs = min(fs[~np.isnan(fs)].min(), min_fs)
        for i in range(fs.shape[0]):
            plt.plot(range(n_iter+1), fs[i, :] - min_fs, label=labels[i], color=colors[i], ls=linestyle[i])
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.xlabel('\# iter')
        plt.ylim([1e-8, (fs[~np.isnan(fs)] - min_fs).max()])
        plt.ylabel(r'$R^{(i)} - R_*$')
        plt.savefig(f'imgs/{args.setting}_{args.dset}.pdf')
        if not args.dont_show:
            plt.show()
