#%% Imports

import numpy as np
from sklearn import datasets

#%% Load dataset
X, y = datasets.load_diabetes(return_X_y=True)
ntrain, inpdim = X.shape
# Standardize data (easier to set the l1_ratio parameter)
X -= X.mean(axis=0)
X /= X.std(axis=0)


#%% Define models of interest
p = 2
# compute inpute norm
l2norm = np.mean(np.linalg.norm(X, axis=1, ord=2))
linfnorm = np.mean(np.max(np.abs(X), axis=1))
# Define adversarial radius
norm = linfnorm if p == np.Inf else l2norm
adv_radius = 0.01 * norm


# %% solve l2 problem
import time
from linadvtrain.solvers import lin_advtrain

t = time.time()
adv_radius = 1000
n_iter = 10
params = lin_advtrain(X, y, adv_radius=adv_radius, max_iter=n_iter)
elapsed = time.time() - t
print(f'Elapsed time: {elapsed:4.3e} seconds')

#%%
if __name__ == '__main__':
    from linadvtrain.cvxpy_impl import AdversarialTraining

    mdl = AdversarialTraining(X, y, p=2)
    sol = mdl(adv_radius, verbose=True)

    print(np.linalg.norm(sol - params, ord=2))



