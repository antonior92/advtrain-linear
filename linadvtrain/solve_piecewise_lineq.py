import numpy as np


def pos(x):
    return np.maximum(x, 0)


def solve_quadratic_equation(aa, bb, cc):
    """Solve quadratic equation

    Solve the equation:
        aa * x**2 + bb * x + cc == 0
    """
    delta = bb ** 2 - 4 * aa * cc
    sol1 = (-bb + np.sqrt(delta)) / (2 * aa)
    sol2 = (-bb - np.sqrt(delta)) / (2 * aa)
    return sol1, sol2


def solve_piecewise_lineq(coefs, t, rho=1, delta=1, return_line=False):
    # Sort abs coeficients
    abs_coefs = np.abs(coefs)
    abs_coefs.sort()

    if t > np.sum(abs_coefs):
        raise ValueError('t > np.sum(coefs), there is no solution')
    # Get all the picewise segments.
    # Find the piecewise segments of the function
    #     h(x) = sum_i(b[i] - rho * x)_+
    # These piecewise segments define the affine functions
    #     fi(x) = - rho * m[i] * x + c[i]
    m = np.arange(len(abs_coefs))[::-1] + 1
    c = np.cumsum(abs_coefs[::-1])[::-1]

    # break points b[i]
    b = abs_coefs / rho
    # Let g(x) = delta * t + x.
    # Evaluate  the functions fi and g in  in each breakpoint
    # Find the fist value for each fi(b[i]) < g(b[i]) and b[i]
    # the intersection will happen in the line segment corresponding to it
    index = np.sum((- rho * m * b + c) > (delta * b + t))
    # solve the equation (-rho * m[i] * x + c[i]) = (delta * x + t)
    if index < len(b):
        s = (c[index] - t) / (rho * m[index] + delta)
    else:
        s = - t
    if return_line:
        return s, m[index], c[index]
    else:
        return s


def compute_lhs(coefs, s, rho=1):
    s = np.atleast_1d(s)
    coefs = np.atleast_1d(coefs)
    lhs = pos(np.abs(coefs[None, :]) - rho * s[:, None]).sum(axis=1)
    return lhs


def compute_rhs(t, s, delta=1):
    s = np.atleast_1d(s)
    rhs = t + delta * s
    return rhs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    coefs = np.array([0.2, 0.3, 0.4])
    delta = 0.5
    rho = 2

    b = coefs
    t = 0.01

    l = np.linspace(0, max([max(b), -t]), 50)

    s, m, c = solve_piecewise_lineq(coefs, t, rho, delta, return_line=True)

    plt.plot(l, compute_lhs(coefs, l, rho))
    plt.plot(l, compute_rhs(t, l, delta))
    plt.plot(s, t + delta * s , 's')
    plt.plot(l, -rho * m * l + c,)
    plt.ylim(min(-0.1, 1.1*t), 1.5 * sum(np.abs(coefs)))
    plt.show()