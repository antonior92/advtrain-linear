import numpy as np


def identity(x):
    return x


def gd(w0, compute_grad, prox=None, max_iter=100000, lr=1, momentum=0.0, nesterov=False, utol=1e-12, callback=None, every_ith=1):
    if prox is None:
        prox = identity
    w = w0
    new_w = np.copy(w)
    for i in range(max_iter):
        update_param = new_w - w
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







