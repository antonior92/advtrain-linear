# %% Imports
import numpy as np
from linadvtrain.cvxpy_impl import compute_q
from linadvtrain.solve_piecewise_lineq import solve_piecewise_lineq, pos
from linadvtrain.first_order_methods import gd, agd, sgd, saga, gd_with_backtrack, agd_with_backtrack
from linadvtrain.regression import get_radius
def soft_threshold(x, threshold):
    return np.sign(x) * pos(np.abs(x) - threshold)

def split_params(params):
    return params[:-1], params[-1]

def merge_params(w, t):
    return np.hstack([w, t])


#  Implement gradient descent in adversarial training
def projection(param, p=2, rho=1, delta=1):
    """Euclidean projection into the set {(param, max_norm) | delta * ||param||_q <=  rho * max_norm}

    The solution to the optimization problem:
        min_(x,t) ||param - x||_2^2  + (max_norm - t)^2  s.t. delta * ||x||_q <=  rho * t
    """
    param, max_norm = split_params(param)
    norm_dual = np.linalg.norm(param, ord=compute_q(p))
    if delta * norm_dual > rho * max_norm:
        if p == 2:
            new_max_norm = delta * (rho * norm_dual + delta * max_norm) / (delta**2 + rho**2)
            new_param = param * (rho * new_max_norm) / (delta * norm_dual)
            return merge_params(new_param, np.abs(new_max_norm))
        elif p == np.inf:
            threshold = solve_piecewise_lineq(param, max_norm, delta=delta, rho=rho)
            return merge_params(soft_threshold(param, threshold * delta), max_norm + rho * threshold)
    else:
        return merge_params(param, max_norm)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CostFunction:
    def __init__(self, X, y, adv_radius, p):
        self.V = (2 * y - 1)[:, None] * X  # convert y to +1 or - 1
        self.n_train, self.n_params = self.V.shape
        self.adv_radius = adv_radius
        self.q = compute_q(p)
        self.n_params = self.V.shape[1]

    def compute_aux(self, V, w, t):
        return sigmoid(V @ w - self.adv_radius * t)

    def compute_cost(self, params):
        w, t = split_params(params)
        aux = self.compute_aux(self.V, w, t)
        return 1 / self.n_train * np.sum(-np.log(aux))

    def compute_grad(self, params, indexes=None):
        w, t = split_params(params)
        if indexes is None:
            Vi = self.V
        else:
            indexes = np.random.permutation(indexes)
            Vi = self.V[indexes, :]
        aux = self.compute_aux(Vi, w, t)
        grad_param = - 1 / self.n_train * Vi.T @ (1 - aux)
        grad_max_norm = 1 / self.n_train * self.adv_radius * np.sum(1 - aux)
        return merge_params(grad_param, grad_max_norm)

    def compute_jac(self, params, indexes=None):
        w, t = split_params(params)
        if indexes is None:
            Vi = self.V
        else:
            Vi = self.V[indexes, :]
        aux = self.compute_aux(Vi, w, t)
        jac_param = - (Vi.T * (1 - aux)).T
        jac_max_norm = self.adv_radius * (1 - aux)
        return merge_params(jac_param, jac_max_norm[:, None])


def power_method_covmatr(X, num_iterations: int = 10):
    """Estimate maximum eingenvalue of the empirical covariance matrix of X"""
    # Here A = 1/n X.T X
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    np.random.seed(1)
    v = np.random.rand(X.shape[0])
    n = X.shape[0]

    for _ in range(num_iterations):
        v = 1/n * np.dot(X, np.dot(X.T, v))
        s = np.max(v)
        v = v / s
    return s


def lin_advclasif(X, y, adv_radius=None, p=2, verbose=False, method='gd', backtrack=True, callback=None, lr=None,
                  save_costs=True,  max_iter=1000, **kwargs):
    """Linear adversarial classification """
    if adv_radius is None:
        adv_radius = 'randn_zero'
    if isinstance(adv_radius, str):
        adv_radius = get_radius(X, y, adv_radius, p)
    rho = 1/2 * np.sqrt(power_method_covmatr(X))
    cost = CostFunction(X, y, rho, p)
    prox = lambda x: projection(x, p=p, delta=adv_radius, rho=rho)
    w0 = np.zeros(cost.n_params + 1)
    costs = np.empty(max_iter + 1)
    costs[0] = cost.compute_cost(w0)
    costs[1:] = np.nan


    def new_callback(i, w, f, update_size):
        if verbose:
            print(f'Iteration {i} | update size: {update_size:4.3e} | cost: {f} |')
        if callback is not None:
            callback(i, w, f, update_size)
        if save_costs:
            costs[i+1] = f

    if lr is None:
        L = power_method_covmatr(X)
        lr = 2/L
    elif isinstance(lr, str):
        L = power_method_covmatr(X)
        lr = eval(lr)
    else:
        pass

    if method == 'gd':
        if backtrack:
            w = gd_with_backtrack(w0, cost.compute_cost, cost.compute_grad, prox=prox, callback=new_callback, max_iter=max_iter, **kwargs)
        else:
            w = gd(w0, cost.compute_cost, cost.compute_grad, prox=prox, callback=new_callback, lr=lr, max_iter=max_iter, **kwargs)
    elif method == 'agd':
        if backtrack:
            w = agd_with_backtrack(w0, cost.compute_cost, cost.compute_grad, prox=prox, callback=new_callback, max_iter=max_iter, **kwargs)
        else:
            w = agd(w0, cost.compute_cost, cost.compute_grad, prox=prox, callback=new_callback, lr=lr, max_iter=max_iter, **kwargs)
    elif method == 'sgd':
        n_train = X.shape[0]
        w = sgd(w0, cost.compute_cost, cost.compute_grad, n_train, prox=prox, callback=new_callback, lr=lr, max_iter=max_iter,  **kwargs)
    elif method == 'saga':
        n_train = X.shape[0]
        w = saga(w0, cost.compute_cost, cost.compute_jac, n_train, prox=prox, callback=new_callback, lr=lr, max_iter=max_iter, **kwargs)
    param, t = split_params(w)
    return param, {'costs': costs}


if __name__ == "__main__":
    X = np.diag([1, 2, 3, 4])

    max_norm = np.linalg.norm(X, axis=1) .max()

# TODO:
#   5. [ ] Add condition to verify optimality based on suboptimality