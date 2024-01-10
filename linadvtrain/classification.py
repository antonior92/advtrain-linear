# %% Imports
import numpy as np
from linadvtrain.cvxpy_impl import compute_q
from linadvtrain.solve_piecewise_lineq import solve_piecewise_lineq, pos
from linadvtrain.first_order_methods import gd


def soft_threshold(x, threshold):
    return np.sign(x) * pos(np.abs(x) - threshold)

def split_params(params):
    return params[:-1], params[-1]

def merge_params(w, t):
    return np.hstack([w, t])


#  Implement gradient descent in adversarial training
def projection(param, p=2):
    """Euclidean projection into the set {(param, max_norm) | ||param||_q <= max_norm}

    The solution to the optimization problem:
        min_(x,t) ||param - x||_2^2  + (max_norm - t)^2  s.t. ||x||_q <= t
    """
    param, max_norm = split_params(param)
    norm_dual = np.linalg.norm(param, ord=compute_q(p))
    if norm_dual > max_norm:
        if p == 2:
            new_max_norm = (norm_dual + max_norm) / 2
            new_param = param * new_max_norm / norm_dual
            return merge_params(new_param, np.abs(new_max_norm))
        elif p == np.inf:
            threshold = solve_piecewise_lineq(param, max_norm)
            return merge_params(soft_threshold(param, threshold), max_norm + threshold)

    else:
        return merge_params(param, max_norm)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CostFunction:
    def __init__(self, X, y, adv_radius, p, weight_decay=0.0):
        self.V = (2 * y - 1)[:, None] * X  # convert y to +1 or - 1
        self.n_train, self.n_params = self.V.shape
        self.weight_decay = weight_decay
        self.adv_radius = adv_radius
        self.q = compute_q(p)
        self.n_params = self.V.shape[1]

    def compute_aux(self, w, t):
        return sigmoid(self.V @ w - self.adv_radius * t)

    def compute_cost(self, params):
        w, t = split_params(params)
        aux = self.compute_aux(w, t)
        return 1 / self.n_train * np.sum(-np.log(aux))

    def compute_grad(self, params):
        w, t = split_params(params)
        aux = self.compute_aux(w, t)
        grad_param = - 1 / self.n_train * self.V.T @ (1 - aux) + self.weight_decay * w
        grad_max_norm = 1 / self.n_train * self.adv_radius * np.sum(1 - aux)
        return merge_params(grad_param, grad_max_norm)


def lin_advclasif(X, y, adv_radius=None, p=2, weight_decay=0.0, verbose=False, **kwargs):
    """Linear adversarial classification """
    if adv_radius is None:
        adv_radius = 0.001
    cost = CostFunction(X, y, adv_radius, weight_decay)
    prox = lambda x: projection(x, p=p)
    w0 = np.zeros(cost.n_params + 1)
    if verbose:
        def callback(i, w, update_size):
            print(f'Iteration {i} | update size: {update_size:4.3e} | cost: {cost.compute_cost(w)} | ')
    else:
        callback = None
    w = gd(w0, cost.compute_grad, prox=prox, callback=callback, **kwargs)
    param, t = split_params(w)
    return param, {}



# TODO:
#   1. [x] Add acceleration. It will make it faster and easier to check for discrepancies.
#   2. [x] Refactor and move this to test file...
#   3. [ ] The projection condition is wrong for the linf norm
#   5. [ ] Add condition to verify optimality based on suboptimality
#   4. [ ] Convert the inner loop to  C based.. It will also make it faster