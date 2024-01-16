import numpy as np
from linadvtrain.first_order_methods import gd, sgd, saga, cg
import pytest
from numpy import allclose
from sklearn.linear_model._ridge import _ridge_regression


def test_gd_on_linear_regresion():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 10)
    y = rng.randn(100)

    param = np.linalg.pinv(X) @ y

    def grad(param):
        return X.T @ (X @ param - y)

    p = gd(np.zeros(10), grad, lr=0.001)

    allclose(p, param)



def test_sgd_on_linear_regresion():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 10)
    y = rng.randn(100)

    param = np.linalg.pinv(X) @ y

    def grad(param, indexes):
        return X[indexes, :].T @ (X[indexes, :]  @ param - y[indexes])

    p = sgd(np.zeros(10), grad, 100, lr=0.001, max_iter=100)

    assert allclose(p, param)


def test_sgd_on_linear_regresion_with_minibatches():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 10)
    y = rng.randn(100)

    param = np.linalg.pinv(X) @ y

    def grad(param, indexes):
        Xi = X[indexes, :]
        yi = y[indexes]
        return 1 / X.shape[0] * Xi.T @ (Xi @ param - yi)

    for b in [1, 10, 100]:
        p = sgd(np.zeros(10), grad, 100, batch_size=b, lr=0.001, max_iter=100)
        print(np.linalg.norm(p - param))
    assert allclose(p, param)



def test_saga_on_linear_regresion_with_minibatches():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 10)
    y = rng.randn(100)

    param = np.linalg.pinv(X) @ y

    def compute_jac(param, indexes):
        Xi = X[indexes, :]
        yi = y[indexes]
        JacT = Xi.T * (Xi @ param - yi)
        return JacT.T

    for b in [1, 10, 100]:
        p = saga(np.zeros(10), compute_jac, 100, batch_size=b, lr=0.01, max_iter=100)
        print(np.linalg.norm(p-param))


def test_cg():
    n_params = 10
    rng = np.random.RandomState(1)
    X_aux = rng.randn(100, n_params)
    X = X_aux @ X_aux.T
    y = X_aux @ rng.randn(n_params)
    param_linalg = np.linalg.pinv(X) @ y
    params_cg = cg(X, y)

    assert allclose(param_linalg, params_cg)

def test_cg_ridge():
    n_params = 10
    rng = np.random.RandomState(1)
    X_aux = rng.randn(100, n_params)
    X = X_aux @ X_aux.T
    y = X_aux @ rng.randn(n_params)
    reg = 0.1

    param_ridge = _ridge_regression(X, y, 0.1)

    # primal formulation
    A = X.T @ X + 0.1 * np.eye(100)
    b = X.T @ y
    params_cg = cg(A, b)
    assert allclose(param_ridge, params_cg)

    # Dual formulation
    A = X @ X.T + 0.1 * np.eye(100)
    b = y
    params_cg_dual = X.T @ cg(A, b)
    assert allclose(param_ridge, params_cg_dual)


