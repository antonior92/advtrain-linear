import pytest
import sklearn.datasets
from linadvtrain import lin_advregr
from numpy import allclose
import linadvtrain.cvxpy_impl.regression as cvxpy_impl
import linadvtrain.regression as solvers
import numpy as np


def get_data(n_train=100, n_params=10, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_train, n_params)
    y = rng.randn(n_train)
    return X, y


def get_diabetes():
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    # Standardize data (easier to set the l1_ratio parameter)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X, y


# -------- Test eta trick --------- #
def test_two_parameter():
    w1 = np.array([75, 141, 206, 135])
    w2 = np.array([1, 1, 3, 4])
    c1, c2 = solvers.eta_trick([w1, w2])
    lhs = (w1 + w2) ** 2
    rhs = c1 * w1 ** 2 + c2 * w2 ** 2
    assert allclose(lhs, rhs)

def test_with_zeros():
    w1 = np.array([75])
    w2 = np.array([0])
    w3 = np.array([4])
    c1, c2, c3 = solvers.eta_trick([w1, w2, w3])
    lhs = (w1 + w2 + w3) ** 2
    rhs = c1 * w1 ** 2 + c3 * w3 ** 2
    assert allclose(lhs, rhs)


def test_three_parameters():
    w1 = np.array([75, 141, 206, 135])
    w2 = np.array([1, 2, 3, 4])
    w3 = np.array([7, 14, 20, 13])
    c1, c2, c3 = solvers.eta_trick([w1, w2, w3])
    lhs = (w1 + w2 + w3) ** 2
    rhs = c1 * w1 ** 2 + c2 * w2 ** 2 + c3 * w3 ** 2
    assert allclose(lhs, rhs)


# -------- Lin Adv train --------- #
@pytest.mark.parametrize("reg", [0.01])
def test_sqlasso(reg):
    # Generate data
    X, y = get_data()
    n_train, n_params = X.shape
    # Test
    params, info = solvers.sq_lasso(X, y, reg=reg, verbose=False, max_iter=1000)
    assert params.shape == (n_params,)

    mdl = cvxpy_impl.SqLasso(X, y)
    params_cvxpy = mdl(reg)
    assert allclose(params_cvxpy, params)


# -------- Lin Adv train --------- #
@pytest.mark.parametrize("adv_radius", [0.1, 0.01, 0.001])
def test_l2(adv_radius):
    # Generate data
    X, y = get_data()
    n_train, n_params = X.shape
    # Test dimension
    params, info = lin_advregr(X, y, adv_radius=adv_radius, verbose=False, p=2, method='w-ridge')
    assert params.shape == (n_params,)

    # Compare with cvxpy
    mdl = cvxpy_impl.AdversarialTraining(X, y, p=2)
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    assert allclose(params_cvxpy, params,  rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize("adv_radius",  [0.01, 0.1])
@pytest.mark.parametrize("method", ['w-ridge', 'w-sqlasso'])
def test_l1(adv_radius, method):
    # Generate data
    X, y = get_data()
    n_train, n_params = X.shape
    # Test dimension
    params, info = lin_advregr(X, y, adv_radius=adv_radius, verbose=False, utol=1e-200, max_iter=2000, p=np.inf, method=method)
    assert params.shape == (n_params,)

    # Compare with cvxpy
    mdl = cvxpy_impl.AdversarialTraining(X, y, p=np.inf)
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    print(np.linalg.norm(params_cvxpy - params))
    print(params_cvxpy, params)
    assert allclose(params_cvxpy, params,  rtol=1e-8, atol=1e-8)


# Case where w_params would go to infinity
@pytest.mark.parametrize("adv_radius",  [0.005478901179593945])
def test_l1_diabetes(adv_radius):
    # Generate data
    X, y = get_diabetes()
    n_train, n_params = X.shape
    # Test dimension
    params, info = lin_advregr(X, y, adv_radius=adv_radius, verbose=False, p=np.inf, method='w-sqlasso')
    assert params.shape == (n_params,)

    # Compare with cvxpy
    mdl = cvxpy_impl.AdversarialTraining(X, y, p=np.inf)
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    assert allclose(params_cvxpy, params,  rtol=1e-8, atol=1e-8)


# Case where w_params would go to infinity
def test_l1_diabetes_zero():
    # Generate data
    X, y = get_diabetes()
    n_train, n_params = X.shape
    # Test if adv_radius='zeros' produce zero solution
    params, info = lin_advregr(X, y, adv_radius='zero', p=np.inf, method='w-ridge', max_iter=10000)
    assert allclose(params, np.zeros_like(params),  atol=1e-2)  # note: the convergence seems to be slow here, that is why the high atol

    # Test if adv_radius>solvers.get_radius(X, y, 'zero', p=p) produce zero solution
    adv_radius = solvers.get_radius(X, y, 'zero', p=np.inf) + 0.1
    params, info = lin_advregr(X, y, adv_radius=adv_radius, p=np.inf, max_iter=1000)
    assert allclose(params, np.zeros_like(params),  rtol=1e-8, atol=1e-8)

    # Test if adv_radius < solvers.get_radius(X, y, 'zero', p=p) produce non-zero solution
    adv_radius = solvers.get_radius(X, y, 'zero', p=np.inf) - 0.1
    params, info = lin_advregr(X, y, adv_radius=adv_radius, p=np.inf, max_iter=1000)
    assert not allclose(params, np.zeros_like(params),  rtol=1e-8, atol=1e-8)



if __name__ == '__main__':
    pytest.main()

