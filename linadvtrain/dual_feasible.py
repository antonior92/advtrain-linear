""""Routines for finding dual feasible point

\|A.T a\| <= \delta sum(a)

"""
import numpy as np
import cvxpy as cp


class DualFeasible(object):
    def __init__(self, A, pnorm):
        self.A = A
        self.pnorm = pnorm

    def minimum_delta(self, a):
        num = np.linalg.norm(self.A.T @ a, ord=self.pnorm, keepdims=True, axis=0)
        den = np.sum(a, keepdims=True, axis=0)
        return num / den

    def solve_l2_value(self, **kwargs):
        n = self.A.shape[0]
        dualp = cp.Variable(n)
        normp = cp.quad_form(dualp, (self.A @ self.A.T),  assume_PSD=True)
        vsum = cp.sum(dualp)

        prob = cp.Problem(cp.Minimize(normp), [vsum == 1, dualp >= np.zeros(n)])
        print(f'Is qp = {prob.is_qp()}')
        prob.solve(**kwargs, )

        return dualp.value

    def solve_linf_value(self, **kwargs):
        n = self.A.shape[0]
        dualp = cp.Variable(n)
        r =  cp.Variable(1)
        aux = self.A.T @ dualp
        vsum = cp.sum(dualp)
        prob = cp.Problem(cp.Minimize(r), [vsum == 1, dualp >= np.zeros(n), aux <= r, -aux <= r])

        print(f'Is qp = {prob.is_qp()}')
        prob.solve(**kwargs, )

        return dualp.value