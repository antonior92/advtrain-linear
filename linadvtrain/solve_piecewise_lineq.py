import numpy as np
from numba import njit

def pos(x):
    return np.maximum(x, 0)


def solve_piecewise_lineq(coefs, t, delta=1, rho=1):
    # Sort abs coeficients
    abs_coefs = np.abs(coefs)
    abs_coefs.sort()

    if rho * t > delta * np.sum(abs_coefs):
        raise ValueError('t > np.sum(coefs), there is no solution')
    # Get all the picewise segments.
    # Find the piecewise segments of the function
    #     h(x) = sum_i(b[i] - delta * x)_+
    # These piecewise segments define the affine functions
    #     fi(x) = -delta * m[i] * x + c[i]
    m = np.arange(len(abs_coefs))[::-1] + 1
    c = np.cumsum(abs_coefs[::-1])[::-1]

    # break points b[i]
    b = abs_coefs / delta
    # Let g(x) = delta * t + x.
    # Evaluate  the functions fi and g in  in each breakpoint
    # Find the fist value for each fi(b[i]) < g(b[i]) and b[i]
    # the intersection will happen in the line segment corresponding to it
    index = np.sum((- delta * m * b + c) > (rho * rho / delta) * b + (rho / delta) * t)
    # solve the equation (-delta * m[i] * x + c[i]) = ((rho * rho/ delta) * x + (rho / delta) * t)
    if index < len(b):
        s = (c[index] - (rho / delta) * t) / (delta * m[index] + rho * rho / delta)
        return s, m[index], c[index]
    else:
        s = - t
        return s, 0, 0



def compute_lhs(coefs, s, delta=1):
    s = np.atleast_1d(s)
    coefs = np.atleast_1d(coefs)
    lhs = pos(np.abs(coefs[None, :]) - delta * s[:, None]).sum(axis=1)
    return lhs


def compute_rhs(t, s, rho=1, delta=1):
    s = np.atleast_1d(s)
    rhs = (rho / delta) * t + rho * rho / delta * s
    return rhs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    coefs = np.array([0.2, 0.3, 0.4])
    delta = 2
    rho = 1

    b = coefs
    t = -1

    l = np.linspace(0, max([max(b), -t]), 50)

    s, m, c = solve_piecewise_lineq(coefs, t, delta, rho)

    plt.plot(l, compute_lhs(coefs, l, delta))
    plt.plot(l, compute_rhs(t, l, rho, delta))
    plt.plot(s, (rho / delta) * s + (rho * rho / delta) * t , 's')
    plt.plot(l, -delta * m * l + c,)
    plt.ylim(min(-0.1, 1.1*t), 1.5 * sum(np.abs(coefs)))
    plt.show()