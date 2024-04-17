import pytest
import sklearn.datasets
from linadvtrain import lin_advregr
from numpy import allclose
from linadvtrain.solve_piecewise_lineq import solve_piecewise_lineq, compute_lhs, compute_rhs
import numpy as np


@pytest.mark.parametrize("coefs", [
    [0.2, 0.3, 0.4],
    [-0.2, 3, -2],
    [-1, -2, -3],
    [-10, -2, -0.5],
    [1, 2, 3]
])
@pytest.mark.parametrize("rho", [1, 2])
@pytest.mark.parametrize("delta", [1, 2, 3])
@pytest.mark.parametrize("perc_t", [-0.1, 0, 0.1, 0.7, 0.8])
def test_solvepiecewise_lineq(coefs, perc_t, rho, delta):
    t = delta/rho * perc_t * np.sum(np.abs(coefs))
    s, _m, _m = solve_piecewise_lineq(coefs, t, rho=rho, delta=delta)
    assert (np.allclose(compute_lhs(coefs, s, delta=delta), compute_rhs(t, s, rho=rho, delta=delta)))
