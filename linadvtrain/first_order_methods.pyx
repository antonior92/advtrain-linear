import cython
import numpy as np
cimport numpy as np


def identity(np.ndarray[np.float64_t, ndim=1]  x):
    return x

def gd(np.ndarray[np.float64_t, ndim=1] w0,  object compute_grad,  object prox = None,
      object callback=None,  int max_iter=10000, float lr=1.0, nesterov=False,
       float momentum=0.0,  float utol=1e-12, int every_ith=1):
    if prox is None:
      prox = identity

    cdef np.ndarray[np.float64_t, ndim = 1] w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] new_w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] grad = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] update_param = np.zeros_like(w0)
    cdef int i

    for i in range(max_iter):
        if nesterov:
            grad = compute_grad(w + momentum * update_param)
        else:
            grad = compute_grad(w)
        # Compute update using momentum (for momentum = 0 we just recover gd)
        update_param = momentum * update_param - lr * grad
        # Do updates
        new_w = prox(w + update_param)
        update_size = np.linalg.norm(new_w - w)
        w = new_w
        if i % every_ith == 0 and callback is not None:
            callback(i, w, update_size)
        if update_size < utol:
            break
    return w



def sgd(np.ndarray[np.float64_t, ndim=1] w0,  object compute_grad,  int n_train, int batch_size=1,
        object prox = None, object callback=None,  int max_iter=100, float lr=1.0, nesterov=False,
        float momentum=0.0,  float utol=1e-12, int every_ith=1):
    if prox is None:
       prox = identity

    cdef np.ndarray[np.float64_t, ndim = 1] w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] new_w = np.copy(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] grad = np.zeros_like(w0)
    cdef np.ndarray[np.float64_t, ndim = 1] update_param = np.zeros_like(w0)
    cdef np.ndarray[long, ndim = 1]  indexes = np.random.permutation(np.arange(n_train))
    cdef np.ndarray[long, ndim = 1]  indexes_batch = np.zeros(batch_size, dtype=long)
    cdef int i, s
    cdef int n_batches = n_train // batch_size

    for i in range(max_iter):
        for s in range(n_batches):
            indexes_batch[:] = indexes[batch_size*s:batch_size*(s+1)]
            if nesterov:
                grad = compute_grad(w + momentum * update_param, indexes_batch)
            else:
                grad = compute_grad(w, indexes_batch)
            # Compute update using momentum (for momentum = 0 we just recover gd)
            update_param = momentum * update_param - lr * grad
            # Do updates
            new_w = prox(w + update_param)
            update_size = np.linalg.norm(new_w - w)
            w = new_w
        if i % every_ith == 0 and callback is not None:
            callback(i, w, update_size)
        if update_size < utol:
            break
        indexes = np.random.permutation(np.arange(n_train))
    return w


def saga(np.ndarray[np.float64_t, ndim=1] w0, object compute_jac, int n_train, int batch_size=1,
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
            callback(i, w, update_size)
        if update_size < utol:
            break
        indexes = np.random.permutation(np.arange(n_train))
    return w

def cg(object X, np.ndarray[np.float64_t, ndim=1] y,
       np.ndarray[np.float64_t, ndim=1] param0=None,
       int max_iter=0, object precond=None, float tol=1e-8):
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
    cdef np.ndarray[np.float64_t, ndim=1] update_d = -np.copy(resid)
    cdef np.ndarray[np.float64_t, ndim=1] precond_resid = precond(resid)
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
        if resid_sqnorm_next < 1e-10:
            break
        beta = resid_sqnorm_next / resid_sqnorm
        update_d = - precond_resid + beta * update_d
        resid_sqnorm = resid_sqnorm_next

    return param