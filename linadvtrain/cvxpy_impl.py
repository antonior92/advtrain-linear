# Import packages.
import cvxpy as cp
import numpy as np


def compute_q(p):
    if p != np.Inf and p > 1:
        q = p / (p - 1)
    elif p == 1:
        q = np.Inf
    else:
        q = 1
    return q


class AdversarialTraining:
    def __init__(self, X, y, p):
        m, n = X.shape
        q = compute_q(p)
        # Formulate problem
        param = cp.Variable(n)
        param_norm = cp.pnorm(param, p=q)
        adv_radius = cp.Parameter(name='adv_radius', nonneg=True)
        abs_error = cp.abs(X @ param - y)
        adv_loss = 1 / m * cp.sum((abs_error + adv_radius * param_norm) ** 2)
        prob = cp.Problem(cp.Minimize(adv_loss))
        self.prob = prob
        self.adv_radius = adv_radius
        self.param = param
        self.warm_start = False

    def __call__(self, adv_radius, **kwargs):
        try:
            self.adv_radius.value = adv_radius
            self.prob.solve(warm_start=self.warm_start, **kwargs)
            v = self.param.value
        except:
            v = np.zeros(self.param.shape)
        return v


class SqLasso:
    def __init__(self, X, y):
        nrows, ncols = X.shape
        # Formulate problem
        param = cp.Variable(ncols)
        param_norm = cp.pnorm(param, p=1)
        reg = cp.Parameter(name='reg', nonneg=True)
        adv_loss = cp.sum((X @ param - y) ** 2) + reg * param_norm ** 2
        prob = cp.Problem(cp.Minimize(adv_loss))
        self.prob = prob
        self.reg = reg
        self.param = param
        self.warm_start = False

    def __call__(self, reg, **kwargs):
        try:
            self.reg.value = reg
            self.prob.solve(warm_start=self.warm_start, **kwargs)
            v = self.param.value
        except:
            v = np.zeros(self.param.shape)
        return v




class MinimumNorm():
    def __init__(self, X, y, p, **kwargs):
        ntrain, nfeatures = X.shape

        param = cp.Variable(nfeatures)
        objective = cp.Minimize(cp.pnorm(param, p=p))
        constraints = [y == X @ param, ]
        prob = cp.Problem(objective, constraints)

        try:
            result = prob.solve(**kwargs)
            self.param = param.value
            self.alpha = constraints[0].dual_value
        except:
            self.param = np.zeros(nfeatures)
            self.alpha = np.zeros(ntrain)
        self.prob = prob
        self.ntrain = ntrain

    def __call__(self):
        return self.param

    def adv_radius(self):
        return 1 / (self.ntrain * np.max(np.abs(self.alpha)))
