# %% Imports
from sklearn import datasets
import matplotlib
matplotlib.use('webagg')
import numpy as np
from linadvtrain.dual_feasible import DualFeasible


def get_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X -= np.mean(X, axis=0)
    X /= X.max(axis=0)  # Normalize each feature to be in [-1, 1], so adversarial training is fair
    y = np.asarray(y, dtype=np.float64)
    return X, y


def test_check_dual_feasible():
    X, y = get_data()
    A = (2 * y[:, None] - 1) * X
    df = DualFeasible(A, pnorm=2)

    v = np.ones(A.shape[0])

    assert df.minimum_delta(v) > 0   # Check it works for alphas_i = 1
    assert np.isnan(df.minimum_delta(np.zeros(A.shape[0])))   # Check if gives nan for 0


def test_check_dual_feasible_multiplepoints():
    X, y = get_data()
    A = (2 * y[:, None] - 1) * X
    df = DualFeasible(A, pnorm=2)

    v = np.ones((A.shape[0], 3))

    assert df.minimum_delta(v).size == 3
    assert (df.minimum_delta(v) > 0).all()   # Check it works for alphas_i = 1
    assert np.isnan(df.minimum_delta(np.zeros(A.shape[0])))   # Check if gives nan for 0


def test_l2solve():
    X, y = get_data()
    A = (2 * y[:, None] - 1) * X

    df = DualFeasible(A, pnorm=2)

    v = df.solve_l2_value(verbose=True)
    for d in (df.minimum_delta(v) * np.logspace(0.00001, 2, 10)):
        assert np.linalg.norm(A.T @ v, ord = 2) < d * np.sum(v)


def test_linfsolve():
    X, y = get_data()
    A = (2 * y[:, None] - 1) * X

    df = DualFeasible(A, pnorm=2)

    v = df.solve_linf_value(verbose=True)
    print(df.minimum_delta(v))
    for d in (df.minimum_delta(v) * np.logspace(0.00001, 2, 10)):
        assert np.linalg.norm(A.T @ v, ord=np.inf) < d * np.sum(v)




