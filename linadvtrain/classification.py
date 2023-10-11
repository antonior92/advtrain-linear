# %% Imports
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import linear_model
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('webagg')
import tqdm
from linadvtrain.cvxpy_impl import compute_q
from linadvtrain.solve_piecewise_lineq import solve_piecewise_lineq, pos


def soft_threshold(x, threshold):
    return np.sign(x) * pos(np.abs(x) - threshold)


#  Implement gradient descent in adversarial training
def projection(param, max_norm, p=2):
    """Euclidean projection into the set {(param, max_norm) | ||param||_q <= max_norm}

    The solution to the optimization problem:
        min_(x,t) ||param - x||_2^2  + (max_norm - t)^2  s.t. ||x||_q <= t
    """
    norm_dual = np.linalg.norm(param, ord=compute_q(p))
    if norm_dual > max_norm:
        if p == 2:
            new_max_norm = (norm_dual + max_norm) / 2
            new_param = param * new_max_norm / norm_dual
            return new_param, np.abs(new_max_norm)
        elif p == np.inf:
            threshold = solve_piecewise_lineq(param, max_norm)
            return soft_threshold(param, threshold), max_norm / (1 - threshold)

    else:
        return param, max_norm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CostFunction:
    def __init__(self, X, y, adv_radius, p, weight_decay=0.0):
        self.V = (2 * y - 1)[:, None] * X  # convert y to +1 or - 1
        self.n_train, self.n_params = self.V.shape
        self.weight_decay = weight_decay
        self.adv_radius = adv_radius
        self.q = compute_q(p)

    def compute_aux(self, w, t=None):
        if t is None:
            t = np.linalg.norm(w, ord=self.q)
        return sigmoid(self.V @ w - self.adv_radius * t)

    def compute_cost(self, w, t=None):
        aux = self.compute_aux(w, t)
        return 1 / self.n_train * np.sum(-np.log(aux))

    def compute_grad(self, w, t=None):
        aux = self.compute_aux(w, t)
        grad_param = - 1 / self.n_train * self.V.T @ (1 - aux) + self.weight_decay * w
        if t is None:
            return grad_param
        else:
            grad_max_norm = 1 / self.n_train * self.adv_radius * np.sum(1 - aux)
            return grad_param, grad_max_norm


def lin_advclasif(X, y, adv_radius=None, max_iter=100000, verbose=False,
                  p=2, utol=1e-12, lr=1, momentum=0.0, nesterov=False,
                  weight_decay=0.0):
    """Linear adversarial classification """
    if adv_radius is None:
        adv_radius = 0.001
    cost = CostFunction(X, y, adv_radius, weight_decay)
    w = np.zeros(cost.n_params)
    t = 0

    new_w, new_t = np.copy(w), t
    for i in range(max_iter):
        update_param = new_w - w
        update_max_norm = new_t - t
        if nesterov:
            grad_param, grad_max_norm = cost.compute_grad(w + momentum * update_param, t + momentum * update_max_norm)
        else:
            grad_param, grad_max_norm = cost.compute_grad(w, t)
        # Compute update using momentum (for momentum = 0 we just recover gd)
        update_param = momentum * update_param - lr * grad_param
        update_max_norm = momentum * update_max_norm - lr * grad_max_norm
        new_w, new_t = projection(w + update_param, t + update_max_norm, p=p)
        update_size = np.sqrt(np.linalg.norm(new_w - w) ** 2 + np.linalg.norm(new_t - t) ** 2)
        if verbose and i % 1000 == 0:
            print(np.linalg.norm(new_w - w))
            print(np.linalg.norm(new_t - t))
            print(f'Iteration {i} | update size: {update_size:4.3e} | cost: {cost.compute_cost(new_w, new_t)} | max_norm: {new_t:4.3e} | ')
        w = new_w
        t = new_t
        if update_size < utol:
            break
        # TODO: add stop criteria based on gradient?
    return w, {}



# TODO:
#   1. [x] Add acceleration. It will make it faster and easier to check for discrepancies.
#   2. [x] Refactor and move this to test file...
#   3. [ ] The projection condition is wrong for the linf norm
#   5. [ ] Add condition to verify optimality based on gradient (it could be useful to check for multiple solutions, and compare the solutions when they are different)
#   4. [ ] Convert the inner loop to  C based.. It will also make it faster