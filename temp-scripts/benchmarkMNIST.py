import matplotlib as mpl
import matplotlib.pyplot as plt
from linadvtrain.classification import lin_advclasif, CostFunction
import numpy as np
import linadvtrain.cvxpy_impl as cvxpy_impl
import time
from sklearn.datasets import fetch_openml


X, y = fetch_openml('mnist_784', parser='auto', return_X_y=True)
X = X.values.astype(np.float64)
X /= 255  # Normalize between [0, 1]
y = np.asarray(y == '9', dtype=np.float64)



if __name__ == "__main__":
    import cProfile

    #lin_advclasif(X, y, adv_radius=0.01, verbose=True, p=2, max_iter=2, method='saga')
    cProfile.run("lin_advclasif(X, y, adv_radius=0.01, verbose=True, p=np.inf, max_iter=2, method='saga')", sort=1)
    cProfile.run("lin_advclasif(X, y, adv_radius=0.01, verbose=True, p=np.inf, max_iter=2, method='gd')", sort=1)

