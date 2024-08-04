import cython
import numpy as np
cimport numpy as np


def identity(np.ndarray[np.float64_t, ndim=1]  x):
    return x

def gd(np.ndarray[np.float64_t, ndim=1] w0,  object compute_cost, object compute_grad,
       object prox = None,  object callback=None,  int max_iter=10000, float lr=1.0, object decreasing_lr = False,
       float utol=1e-12, int every_ith=1):
    if prox is None:
      prox = identity

    cdef np.ndarray[np.float64_t, ndim = 1] w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] new_w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] grad = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] update_param = np.zeros_like(w0)
    cdef int i

    for i in range(max_iter):
        cost = compute_cost(w)
        grad = compute_grad(w)
        # Do updates
        if decreasing_lr:
            lr_ =  lr / np.sqrt(i+1)
        else:
            lr_ = lr
        new_w = prox(w - lr_ * grad)
        update_size = np.linalg.norm(new_w - w)
        w = new_w
        if i % every_ith == 0 and callback is not None:
            callback(i, w, cost, update_size)
        if update_size < utol:
            break
    return w


def gd_with_backtrack(np.ndarray[np.float64_t, ndim=1] w0, object compute_cost, object compute_grad,
                      object prox = None,  object callback=None,  int max_iter=10000, float max_lr=1e10,
                      float decreasing_factor=1.5, float utol=1e-12, int every_ith=1):
    if prox is None:
      prox = identity

    cdef np.ndarray[np.float64_t, ndim = 1] w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] new_w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] grad = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] update_param = np.zeros_like(w0)
    cdef float f = compute_cost(w0)
    cdef float next_f
    cdef float quadratic_approx_f
    cdef float lr = max_lr
    cdef int MAXBACKTRACK = 100
    cdef int i, j

    for i in range(max_iter):
        grad = compute_grad(w)
        # Backtrack line search
        for j in range(MAXBACKTRACK):
            new_w = prox(w - lr * grad)
            next_f = compute_cost(new_w)
            update_size = np.linalg.norm(new_w - w)
            quadratic_approx_f = f + grad.dot(new_w - w) + 1/(2 * lr) * update_size * update_size
            if next_f <= quadratic_approx_f:
                f = next_f
                w = new_w
                break
            else:
                lr = lr / decreasing_factor
        if i % every_ith == 0 and callback is not None:
            callback(i, w, f, update_size)
        if update_size < utol:
            break
    return w




def agd(np.ndarray[np.float64_t, ndim=1] w0,  object compute_cost, object compute_grad,  object prox = None,
        object callback=None,  int max_iter=10000, float lr=1.0, float momentum = 1.0,
        float utol=1e-12, int every_ith=1):
    if prox is None:
      prox = identity

    cdef np.ndarray[np.float64_t, ndim = 1] w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] new_w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] grad = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] look_ahead_w = np.copy(w0)
    cdef int i
    cdef float ti = 1
    cdef float ti_next = 1

    for i in range(max_iter):
        # Do updates
        new_w = prox(look_ahead_w  - lr * compute_grad(look_ahead_w))
        ti_next = (1 + np.sqrt(1 + 4 * ti * ti)) / 2
        look_ahead_w  = new_w  + momentum * (ti - 1) / ti_next * (new_w - w)
        update_size = np.linalg.norm(new_w - w)
        w = new_w
        f = compute_cost(w)
        ti = ti_next
        if i % every_ith == 0 and callback is not None:
            callback(i, w, f, update_size)
        if update_size < utol:
            break
    return w




def agd_with_backtrack(np.ndarray[np.float64_t, ndim=1] w0, object compute_cost,  object compute_grad,  object prox = None,
        object callback=None,  int max_iter=10000, float momentum = 1.0, float max_lr=1e10,
                      float decreasing_factor=1.5, float utol=1e-12, int every_ith=1):
    if prox is None:
      prox = identity

    cdef np.ndarray[np.float64_t, ndim = 1] w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] new_w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] grad = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] look_ahead_w = np.copy(w0)
    cdef int i
    cdef float ti = 1
    cdef float ti_next = 1
    cdef float f = compute_cost(w0)
    cdef float next_f
    cdef float quadratic_approx_f
    cdef float lr = max_lr
    cdef int MAXBACKTRACK = 100

    for i in range(max_iter):
        # Backtrack line search
        f = compute_cost(look_ahead_w)
        grad = compute_grad(look_ahead_w)
        for j in range(MAXBACKTRACK):
            new_w = prox(look_ahead_w  - lr * grad)
            next_f = compute_cost(new_w)
            update_size = np.linalg.norm(new_w - look_ahead_w)
            quadratic_approx_f = f + grad.dot(new_w - look_ahead_w) + 1 / (2 * lr) * update_size * update_size
            if next_f <= quadratic_approx_f:
                break
            else:
                lr = lr / decreasing_factor
        ti_next = (1 + np.sqrt(1 + 4 * ti * ti)) / 2
        look_ahead_w  = new_w  + momentum * (ti - 1) / ti_next * (new_w - w)
        update_size = np.linalg.norm(new_w - w)
        w = new_w
        f = next_f
        ti = ti_next
        if i % every_ith == 0 and callback is not None:
            callback(i, w, f, update_size)
        if update_size < utol:
            break
    return w



def sgd(np.ndarray[np.float64_t, ndim=1] w0,  object compute_cost, object compute_grad,  int n_train, int batch_size=1,
        object prox = None, object callback=None,  int max_iter=100, float lr=1.0, object reduce_lr = True,
        float utol=1e-12, int every_ith=1):
    print('sgd1')
    if prox is None:
       prox = identity

    cdef np.ndarray[np.float64_t, ndim = 1] w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] new_w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] mean_w = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] grad = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] update_param = np.zeros_like(w0)
    cdef np.ndarray[long, ndim = 1]  indexes = np.random.permutation(np.arange(n_train))
    cdef np.ndarray[long, ndim = 1]  indexes_batch = np.zeros(batch_size, dtype=long)
    cdef int i, s

    n_vals = 0
    for i in range(max_iter):
        s = 0
        while batch_size * s < n_train:
            if batch_size*(s+1) > n_train:
                grad = compute_grad(w, indexes[batch_size * s:])
            else:
                grad = compute_grad(w, indexes[batch_size*s:batch_size*(s+1)])
            lr_p=  lr / np.sqrt(i+1) if reduce_lr else lr
            new_w = prox(w - lr_p  * grad)
            update_size = np.linalg.norm(new_w - w)
            w = new_w
            n_vals += 1
            s+= 1
        if i % every_ith == 0 and callback is not None:
            callback(i, w, compute_cost(w), update_size)
        if update_size < utol:
            break
        indexes = np.random.permutation(np.arange(n_train))
    print(w)
    return w


def saga(np.ndarray[np.float64_t, ndim=1] w0, object compute_cost, object compute_jac, int n_train, int batch_size=1,
         object prox = None, object callback=None, int max_iter=10000, float lr=1.0,
         float momentum=0.0, float utol=1e-12, int every_ith=1):
    if prox is None:
        prox = identity

    cdef np.ndarray[np.float64_t, ndim = 1]  w = np.copy(w0)
    cdef int n_params = len(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] new_w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] grad = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] avg_grad = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 2] jac_old = np.zeros((n_train, n_params))
    cdef np.ndarray[np.float64_t, ndim = 2] jac_new = np.zeros((n_train, n_params))
    cdef np.ndarray[np.float64_t, ndim = 2] jac = np.zeros((n_train, n_params))

    cdef np.ndarray[long, ndim = 1] indexes = np.random.permutation(np.arange(n_train))
    cdef np.ndarray[long, ndim = 1] indexes_batch = np.zeros(batch_size, dtype=long)
    cdef int i, s
    cdef int n_batches = n_train // batch_size
    cdef double update_size  = 0.0

    for i in range(max_iter):
        for s in range(n_batches):
            indexes_batch[:] = indexes[batch_size * s:batch_size * (s + 1)]
            jac_new = compute_jac(w, indexes_batch)
            jac_old = jac[indexes_batch, :]
            update = (1 / batch_size * (jac_new - jac_old).sum(axis=0) + avg_grad)
            avg_grad += 1 / n_train * (jac_new - jac_old).sum(axis=0)
            # Do updates
            new_w = prox(w - lr  * update)
            update_size = np.linalg.norm(new_w - w)
            w = new_w
            jac[indexes_batch, :] = jac_new
        if i % every_ith == 0 and callback is not None:
            callback(i, w, compute_cost(w), update_size)
        if update_size < utol:
            break
        indexes = np.random.permutation(np.arange(n_train))
    return w

def cg(object X, np.ndarray[np.float64_t, ndim=1] y,
       np.ndarray[np.float64_t, ndim=1] param0=None,
       int max_iter=0, object precond=None, float rtol=1e-8):
    """Use conjugate gradient to solve the linear system X @ param = y."""

    n_params = X.shape[1]
    if precond is None:
        precond = identity

    if max_iter == 0:
        max_iter = n_params

    cdef np.ndarray[np.float64_t, ndim=1]  param = np.zeros(n_params)
    if param0 is not None:
        param[:] = param0[:]

    cdef np.ndarray[np.float64_t, ndim=1] resid = X.dot(param) - y
    cdef np.ndarray[np.float64_t, ndim=1] precond_resid = precond(resid)
    cdef np.ndarray[np.float64_t, ndim=1] update_d = -precond_resid
    cdef np.ndarray[np.float64_t, ndim=1] X_update_d = np.zeros(n_params)

    cdef double alpha, beta, resid_sqnorm, resid_sqnorm_next

    resid_sqnorm = resid.T @ precond_resid
    for k in range(max_iter):
        X_update_d = X.dot(update_d)
        alpha = resid_sqnorm / (update_d.T @ X_update_d)
        param = param + alpha * update_d
        resid = resid + alpha * X_update_d
        precond_resid = precond(resid)
        resid_sqnorm_next = resid.T @ precond_resid
        if np.linalg.norm(resid)< rtol:
            break
        beta = resid_sqnorm_next / resid_sqnorm
        update_d = - precond_resid + beta * update_d
        resid_sqnorm = resid_sqnorm_next

    return param