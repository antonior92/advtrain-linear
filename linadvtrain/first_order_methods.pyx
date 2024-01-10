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
        update_param = new_w - w
        grad = compute_grad(w)
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