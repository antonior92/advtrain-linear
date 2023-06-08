import timeit
import numpy as np
import linadvtrain as lat
from linadvtrain import solvers
from scipy.linalg import svd, solve, lstsq
from sklearn.linear_model._ridge import _ridge_regression as ridge


n_train = 100
rng = np.random.RandomState(1)
for n_params in [10, 100, 1000, 10000, 100000, 1000000]:
    print(' ### n_params = {} ###'.format(n_params))
    X = rng.randn(n_train, n_params)
    y = rng.randn(n_train)
    print(' --- ridge svd --- ')
    if n_params < 10000:
        print(timeit.timeit(f"ridge(X, y, 0.01, solver='svd')", number=3, globals=globals()))

    print(' --- ridge cholesky --- ')
    print(timeit.timeit(f"ridge(X, y, 0.01, solver='cholesky')", number=3, globals=globals()))

    print(' --- lstq gelss--- ')
    if n_params < 10000:
        print(timeit.timeit(f"lstsq(X, y,lapack_driver='gelss')", number=3, globals=globals()))

    print(' --- lstq lapack gelsy --- ')
    print(timeit.timeit(f"lstsq(X, y,lapack_driver='gelsy')", number=3, globals=globals()))

    print(' --- lstq lapack gelsy --- ')
    print(timeit.timeit(f"lstsq(X, y,lapack_driver='gelsd')", number=3, globals=globals()))

if __name__ == "__main__":
    import numpy as np

