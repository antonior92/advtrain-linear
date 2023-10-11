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


def solve_piecewise_lineq(coefs, t):
    # Sort abs coeficients
    abs_coefs = np.abs(coefs)
    b = np.sort(abs_coefs)

    if t > np.sum(b):
        raise ValueError('t > np.sum(b), there is no solution')
    # Get all the picewise segments.
    # Find the piecewise segments of the function
    #     h(x) = sum_i(b[i] - x)_+
    # These piecewise segments define the affine functions
    #     fi(x) = -m[i] * x + c[i]
    m = np.arange(len(b))[::-1] + 1
    c = np.cumsum(b[::-1])[::-1]

    # Let g(x) = t / (1-x).
    # Evaluate  the functions fi and g in  in each breakpoint
    # Find the fist value for each fi(b[i]) < g(b[i]) and b[i]
    # the intersection will happen in the line segment corresponding to it
    if t >= 0:
        index = np.sum((-m * b + c > t / (1 - b)) & (b > 0))
    else:
        index=0

    # solve the equation fi(x) = g(x) for i = index.
    # select the smallest solution
    s = solve_quadratic_equation(m[index], -(m[index]+c[index]), c[index]-t)[1]
    return s


def compute_lhs(coefs,  s):
    s = np.atleast_1d(s)
    coefs = np.atleast_1d(coefs)
    lhs = pos(np.abs(coefs[None, :]) - s[:, None]).sum(axis=1)
    return lhs


def compute_rhs(t, s):
    s = np.atleast_1d(s)
    rhs = np.where(s != 1, t / (1 - s), np.Inf)
    return rhs

# TODO: think of the casse t is negative and t == 0? what should we do?
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    coefs = np.array([1, 2, 3])

    b = np.array([1, 2, 4])
    t = - 0.1 * np.sum(b)

    l = np.linspace(0, max(b), 50)

    #s = solve_piecewise_lineq(coefs, t)

    #assert(np.allclose(compute_lhs(coefs, s), compute_rhs(t, s)))

    plt.plot(l, compute_lhs(coefs, l))
    plt.plot(l, compute_rhs(t, l))
    #plt.plot(s, t / (1 - s), 's')
    plt.ylim(-0.1, 1.5* sum(np.abs(coefs)))
    plt.show()