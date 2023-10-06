# %% Imports
from sklearn import datasets
import matplotlib
matplotlib.use('webagg')
from linadvtrain.classification import lin_advclasif, CostFunction
import numpy as np
import linadvtrain.cvxpy_impl as cvxpy_impl
import pytest
from numpy import allclose


def get_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X -= np.mean(X, axis=0)
    X /= X.max(axis=0)  # Normalize each feature to be in [-1, 1], so adversarial training is fair
    y = np.asarray(y, dtype=np.float64)
    return X, y


# -------- Lin Adv train --------- #
@pytest.mark.parametrize("adv_radius", [0.1])
def test_l2(adv_radius):
    # Generate data
    X, y = get_data()
    n_train, n_params = X.shape
    # Test dimension
    params, info = lin_advclasif(X, y, adv_radius=adv_radius, verbose=False, p=2, momentum=0.2, nesterov=True)
    assert params.shape == (n_params,)

    # Compare with cvxpy
    mdl = cvxpy_impl.AdversarialClassification(X, y, p=2)
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    print(params_cvxpy, params)
    assert allclose(params_cvxpy, params, rtol=1e-8, atol=1e-2)

@pytest.mark.parametrize("adv_radius", [0.1])
def test_l2_largemomentum(adv_radius):
    # Generate data
    X, y = get_data()
    n_train, n_params = X.shape
    # Test dimension
    params, info = lin_advclasif(X, y, adv_radius=adv_radius, verbose=True, p=2, momentum=0.8, utol=1e-20, nesterov=True)
    assert params.shape == (n_params,)

    # Compare with cvxpy
    mdl = cvxpy_impl.AdversarialClassification(X, y, p=2)
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    print(params_cvxpy, params)
    assert allclose(params_cvxpy, params, rtol=1e-8, atol=1e-2)

@pytest.mark.parametrize("adv_radius", [0.2])
def test_linf(adv_radius):
    # Generate data
    X, y = get_data()
    p = np.inf
    n_train, n_params = X.shape
    # Test dimension
    params, info = lin_advclasif(X, y, adv_radius=adv_radius, verbose=True, p=p, lr=1, momentum=0.8, utol=1e-20, nesterov=True)
    assert params.shape == (n_params,)

    # Compare with cvxpy
    mdl = cvxpy_impl.AdversarialClassification(X, y, p=p)
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    print(params, params_cvxpy)
    print(np.linalg.norm(params, ord=1))
    print(np.linalg.norm(params_cvxpy, ord=1))

    cost = CostFunction(X, y, adv_radius, p)
    print(cost.compute_cost(params))
    print(cost.compute_cost(params_cvxpy))

    assert allclose(params_cvxpy, params, rtol=1e-8, atol=1e-6)

