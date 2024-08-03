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
mpl.rcParams['figure.subplot.bottom'] = 0.25
mpl.rcParams['figure.subplot.right'] = 0.95
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 22
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['legend.labelspacing'] = 0.15
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.15
mpl.rcParams['xtick.major.pad'] = 7
mpl.rcParams["axes.labelpad"] = 6
plt.rcParams['image.cmap'] = 'gray'


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    # Add argument for plotting
    parser.add_argument('--dset', choices=['breast_cancer', 'MNIST', 'MAGIC_C'], default='breast_cancer')
    parser.add_argument('--setting', choices=['compare_lr', 'acceleration', 'stochastic', 'batch_size'], default='stochastic')
    parser.add_argument('--dont_plot', action='store_true', help='Enable plotting')
    parser.add_argument('--dont_show', action='store_true', help='dont show plot, but maybe save it')
    parser.add_argument('--load_data', action='store_true', help='Enable data loading')
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--xlabel', default='\# iter', help='What I have in the X label')
    parser.add_argument('--ls', nargs='+', default=['-', '-', '-', '-', '-', '-'])
    args = parser.parse_args()

    adv_radius = 0.1


    dset = eval(args.dset)
    X, _x, y, _y = dset()

    compare_with_cvxpy = args.dset in ['breast_cancer', ]  # dont compare for other datasets since they are too big

    # Fig 2(a)
    if args.setting == 'compare_lr':
        configs = [{'method': 'gd', 'backtrack': False, 'lr': '2/L'},
                   {'method': 'gd', 'backtrack': False, 'lr': '10/L'},
                   {'method': 'gd', 'backtrack': False, 'lr': '40/L'},
                   {'method': 'gd', 'backtrack': True}]
        labels = [r'lr = 2/$\lambda_{\mathrm{max}}$', 'lr = 10/$\lambda_{\mathrm{max}}$',
                     'lr = 40/$\lambda_{\mathrm{max}}$', 'Backtrack LS']
    # Fig 2(b)
    elif args.setting == 'acceleration':
        configs = [{'method': 'gd', 'backtrack': True},
                   {'method': 'agd', 'backtrack': True}]
        labels = ['GD', 'AGD']

    # Fig 2(c)
    elif args.setting == 'stochastic':
        configs = [{'method': 'gd', 'backtrack': True},
                   {'method': 'sgd', 'lr': '200/L'},
                   {'method': 'saga'}]
        labels = ['GD', 'SGD', 'SAGA']


    # Rebuttal Fig. 4
    elif args.setting == 'batch_size':
        configs = [{'method': 'sgd', 'batch_size': 1, 'lr': '10/L', 'reduce_lr': False},
                   {'method': 'sgd', 'batch_size': 4, 'lr': '10/L','reduce_lr': False},
                   {'method': 'sgd', 'batch_size': 16, 'lr': '10/L',  'reduce_lr': False},
                   {'method': 'sgd', 'batch_size': 64, 'lr': '10/L', 'reduce_lr': False},
                   {'method': 'gd', 'backtrack': False, 'lr': '10/L'},]
        labels = ['BS=1', 'BS=4', 'BS=16', 'BS=64',  'GD']


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
        fs = np.empty([len(configs), args.n_iter + 1])
        for ll, config in enumerate(configs):
            print(f'config {ll}, {config}')
            start_time = time.time()
            params, info = lin_advclasif(X, y, adv_radius=adv_radius,
                                         verbose=True, p=np.inf, max_iter=args.n_iter, **config)
            print(info)
            exec_time = time.time() - start_time
            fs[ll, :] = info['costs']
            print(ll, exec_time)
        np.savetxt(f'data/{args.setting}_{args.dset}.csv', fs)

    if not args.dont_plot:
        colors = ['b', 'g', 'r', 'm', 'k', 'c']
        linestyle = args.ls
        plt.figure()
        min_fs = min(fs[~np.isnan(fs)].min(), min_fs)
        for i in range(fs.shape[0]):
            plt.plot(range(fs.shape[1]), fs[i, :] - min_fs, label=labels[i], color=colors[i], ls=linestyle[i], lw=2)
        plt.yscale('log')
        plt.legend(loc='upper right', bbox_to_anchor=(1.07, 1.12))
        plt.xlabel(args.xlabel)
        plt.ylim([1e-8, fs[~np.isnan(fs)].max()])
        plt.xlim([-1, np.minimum(args.n_iter, fs.shape[1]) + 10])
        # r'$R^{(i)} - R_*$'
        plt.ylabel('sub-optimality')
        plt.savefig(f'imgs/{args.setting}_{args.dset}.pdf')
        if not args.dont_show:
            plt.show()
