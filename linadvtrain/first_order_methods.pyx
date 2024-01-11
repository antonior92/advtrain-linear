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
            update_param = momentum * update_param - lr / n_train * grad
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
            new_w = prox(w - lr / n_batches * update)
            update_size = np.linalg.norm(new_w - w)
            w = new_w
            jac[indexes_batch, :] = jac_new
        if i % every_ith == 0 and callback is not None:
            callback(i, w, update_size)
        if update_size < utol:
            break
        indexes = np.random.permutation(np.arange(n_train))
    return w