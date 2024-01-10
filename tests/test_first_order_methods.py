import numpy as np
from linadvtrain.first_order_methods import gd
import pytest
from numpy import allclose

def test_gd_on_linear_regresion():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 10)
    y = rng.randn(100)

    param = np.linalg.pinv(X) @ y

    def grad(param):
        return X.T @ (X @ param - y)

    p = gd(np.zeros(10), grad, lr=0.001)

    allclose(p, param)
