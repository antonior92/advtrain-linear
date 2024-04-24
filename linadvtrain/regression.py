import numpy as np
from sklearn.linear_model._ridge import _ridge_regression
from linadvtrain.cvxpy_impl import MinimumNorm
from linadvtrain.first_order_methods import cg
from scipy.sparse.linalg import LinearOperator

def ridge(X, y, reg,  *args, **kwargs):
    """Ridge regression."""
    return _ridge_regression(X, y, reg, *args, **kwargs)


class RidgeCG():
    def __init__(self, X, y):
        n_train, n_params = X.shape
        self.X = X
        self.y = y
        self.n_params = n_params
        self.n_train = n_train

    def __call__(self, params0, reg, w_params=None, w_samples=None, max_iter=1):
        X = self.X
        diag = np.linalg.norm(np.sqrt(w_samples[:, None]) * X, ord=2, axis=0) ** 2
        if w_params is None:
            diag += reg
        else:
            diag += reg * w_params

        def precond(x):
            return x / diag

        def f(param):
            out = X.T @ (w_samples * (X @ param))
            if w_params is None:
                out += reg * param
            else:
                out += reg * w_params * param
            return out

        A = LinearOperator(matvec=f, shape=(self.n_params, self.n_params)) #X.T @ (np.diag(w_samples) @ X) + reg * D  # D = np.diag(w_params) if w_params is not None else np.eye(len(params0))
        b = X.T @ (w_samples * self.y)

        params_cg = cg(A, b, params0, precond=precond, max_iter=max_iter) #np.linalg.solve(A, b) #

        return params_cg, {}


def ridge_cg(X, y, reg,  *args, **kwargs):
    """Ridge regression using conjugate gradient."""
    # Rewrite to avoid multiplying X.T @ X several times
    n_train, n_params = X.shape
    if n_params <= n_train:  # Use primal formulation
        A = X.T @ X + reg * np.eye(n_params)
        b = X.T @ y
        params_cg = cg(A, b)
    else:   # Use dual formualation
        A = X @ X.T + 0.1 * np.eye(n_train)
        b = y
        params_cg = X.T @ cg(A, b)
    return params_cg


def compute_q(p):
    if p != np.Inf and p > 1:
        q = p / (p - 1)
    elif p == 1:
        q = np.Inf
    else:
        q = 1
    return q


class Reweighted():
    """Reweighted solver."""
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, X, y,  *args, w_params=None, w_samples=None, **kwargs):
        # Adjust accordingly to sample weights
        if w_samples is None:
            X_rescaled = X
            y_rescaled = y
        else:
            X_rescaled = X * np.sqrt(w_samples[:, None])
            y_rescaled = y * np.sqrt(w_samples[:])
            mask_rows = np.sqrt(w_samples) > 1e-30
            if (~mask_rows).any():  # Solve reduced problem when w_samples is 0
                X_rescaled = X_rescaled[mask_rows, :].copy()
        # Adjust accordingly to parameter weights
        if w_params is not None:
            mask = np.sqrt(w_params) < 1e30
            X_rescaled = X_rescaled / np.sqrt(w_params)[None, :]
            if (~mask).any():  # Solve reduced problem when w_params is Inf
                # remove certain values)
                X_rescaled = X_rescaled[:, mask].copy()
                w_params = w_params[mask].copy()
        # Solve problem
        results = self.solver(X_rescaled, y_rescaled, *args, **kwargs)
        if type(results) is tuple:
            estim_param, info = results
        else:
            estim_param, info = results, {}
        if w_params is not None:
            estim_param = estim_param / np.sqrt(w_params)
            if (~mask).any():
                aux = np.zeros(mask.shape)
                aux[mask] = estim_param
                estim_param = aux
        return estim_param, info


def eta_trick(values, eps=1e-20):
    """Implement eta trick."""
    values = np.atleast_2d(values)
    # for exact solution use eps=0 so that np.abs(values)
    # this might lead to numerical instabilities tho
    abs_values = np.sqrt(values ** 2 + eps)
    sum_of_values = np.sum(abs_values, axis=0)
    c = sum_of_values / (abs_values)
    return c


def sq_lasso(X, y, reg=0.01, max_iter=100, verbose=False, utol=1e-12, w_params_warmstart=None, solver_params=None):
    n_train, n_params = X.shape
    params = np.zeros(n_params)
    # Initialize problem
    w_params = w_params_warmstart
    if solver_params is None:
        solver_params = {}
    for i in range(max_iter):
        # ------- 1. Solve reweighted ridge regression ------
        params_, subprob_info = Reweighted(ridge)(X, y, reg,  w_params=w_params, **solver_params)

        # ------- 2. Perform eta trick  -------
        abs_error = np.abs(X @ params_ - y)
        w_params = eta_trick(params_[:, None]).flatten()

        # -------  Generate report ------
        update_size = np.linalg.norm(params_ - params, ord=2)
        param_norm = np.linalg.norm(params_, ord=1)
        if verbose == True:
            print(f'Iteration {i} | update size: {update_size:4.3e} | w_params: {np.mean(w_params):4.3e} | param norm: {param_norm:4.3e}')
        info = {'w_params': w_params, 'update_size': update_size, 'n_iter': i}
        params = params_  # update parameters

        # ------- Termination criterion -------
        if update_size < utol:
            break

    return params, info


def get_radius(X, y, option, p):
    """Return the adversarial radius for which the zero solution is optimal.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples,)
        Output data.
    option : {'zero', 'randn_zero', 'interp'}
        what type of radius one is interested in
    Returns
    -------
    adv_radius : float or array-like of shape (n_realizations, )
        Adversarial radius for which the zero solution is optimal.
    """
    if option == 'zero':
        return np.linalg.norm(X.T @ y, ord=p) / np.sum(np.abs(y))
    elif option == 'randn_zero':
        n_realizations = 100
        e = np.random.randn(X.shape[0], n_realizations)
        adv_radius_est = np.mean(np.linalg.norm(X.T @ e, ord=p, axis=0) / np.sum(np.abs(e), axis=0))
        return np.percentile(adv_radius_est, 50)  # return value such that 50% of the realizations will yield zero
    elif option == 'interp':
        min_norm = MinimumNorm(X, y, compute_q(p))
        return min_norm.adv_radius()
    else:
        raise ValueError(f'option {option} not recognized')


def lin_advregr(X, y, adv_radius=None, max_iter=100, verbose=False,
                p=2, method='w-ridge', utol=1e-12, solver_params=None,
                callback=None):
    """Linear adversarial regression.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples,)
        Output data.
    adv_radius : float or {'zero', 'randn_zero', 'interp'}, default=None
        Adversarial radius used for training. If none use 'randn_zero'
    max_iter : int, default=100
        Maximum number of iterations.
    verbose : bool, default=False
        Verbosity.
    p : float, default=2
        Norm used for the adversarial radius. Use p=2 for the l2 norm,
        p=np.inf for the l_inf norm.
    method : {'w-ridge', 'w-sqlasso', 'w-cg'}, default='w-ridge'
        Method used for adversarial training.
    utol : float, default=1e-12
        Tolerance for the update size.
    solver_params : dict, default=None
        Parameters passed to the solver.
    callback: function,None default None
        Function that is called every interation. It has the signature:
            fun(i, params, train_loss)

    Returns
    -------
    params : array-like of shape (n_features, )
        Estimated parameters.
    info : dict
        Dictionary containing information about the training.
    """
    if adv_radius is None:
        adv_radius = 'randn_zero'
    if isinstance(adv_radius, str):
        adv_radius = get_radius(X, y, adv_radius, p)
    n_train, n_params = X.shape
    params = np.zeros(n_params)
    # Initialize problem
    w_samples = 1 / n_train * np.ones(n_train)
    w_params = None
    regul_correction = 1
    # Set info
    info = {}
    subprob_info = {}
    if solver_params is None:
        if method == 'w-sqlasso':
            solver_params = {'utol': 1e-12, 'max_iter': 10}
        elif method == 'w-ridge':
            solver_params = {}
        elif method == 'w-cg':
            ridge_ccg = RidgeCG(X, y)
    for i in range(max_iter):
        # ------- 1. Solve reweighted ridge regression ------
        reg = regul_correction * adv_radius ** 2
        if method == 'w-sqlasso':
            params_, subprob_info = Reweighted(sq_lasso)(X, y, reg, w_samples=w_samples, w_params_warmstart=subprob_info.get('w_params', None), **solver_params)
        elif method == 'w-ridge':
            params_, subprob_info = Reweighted(ridge)(X, y, reg, w_samples=w_samples, w_params=w_params, **solver_params)
        elif method == 'w-cg':
            params_, subprob_info = ridge_ccg(params, reg, w_samples=w_samples, w_params=w_params)

        # ------- 2. Perform eta trick  -------
        abs_error = np.abs(X @ params_ - y)
        q = compute_q(p)
        param_norm = np.linalg.norm(params_, ord=q)
        if p == np.inf and method in ['w-ridge', 'w-cg']:
            M = np.abs([abs_error, *[adv_radius * p * np.ones(n_train) for p in params_]])
            c = eta_trick(M)
            w_params = np.sum(c[1:], axis=1)
            # Fix regularization parameter
            regul_correction = np.max(w_params)
            w_params = w_params / np.max(w_params)
        else:
            M = np.abs([abs_error, adv_radius * param_norm * np.ones(n_train)])
            c = eta_trick(M)
            regul_correction = np.sum(c[1])
        w_samples = c[0]

        # Fix regularization parameter
        regul_correction = regul_correction / np.sum(w_samples)
        w_samples = w_samples / np.sum(w_samples)

        # -------  Generate report ------
        update_size = np.linalg.norm(params_ - params, ord=2)
        if verbose == True:
            mean_regul = np.mean(w_samples) * regul_correction if w_samples is not None else regul_correction
            print(f'Iteration {i} | update size: {update_size:4.3e} | regul: {mean_regul:4.3e} | '
                  f'param norm: {param_norm:4.3e} | mean abs error: {np.mean(abs_error):4.3e} | '
                  f'loss: {np.mean((abs_error + adv_radius * param_norm) ** 2)}')
        if callback is not None:
            callback(i, params_, np.mean((abs_error + adv_radius * param_norm) ** 2))
        info = {'w_params': w_params, 'w_samples': w_samples, 'regul_correction':regul_correction,
                'update_size': update_size, 'n_iter': i}
        params = params_  # update parameters

        # ------- Termination criterion -------
        if update_size < utol:
            break
    return params, info