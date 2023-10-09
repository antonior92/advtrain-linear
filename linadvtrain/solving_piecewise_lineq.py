import numpy as np
import matplotlib.pyplot as plt

b = np.array([1, 2, 4])
pos = lambda x: np.maximum(x, 0)
t = 0.1 * np.sum(b)

m = np.arange(len(b))[::-1] + 1
c = np.cumsum(b[::-1])[::-1]

index = np.sum((-m * b + c > t / (1 - b)) | b < 1)

soc = [m[index], -(m[index]+c[index]), c[index]-t]

def solve_quadratic_equation(aa, bb, cc):
    delta = bb**2 - 4 * aa * cc
    sol1 = (-bb + np.sqrt(delta)) / (2 * aa)
    sol2 = (-bb - np.sqrt(delta)) / (2 * aa)
    return sol1, sol2

s = solve_quadratic_equation(*soc)[1]

if __name__ == "__main__":

    l = np.linspace(0, max(b), 50)
    ll = np.linspace(0, 0.9, 50)
    lhs = pos(b[None, :] - l[:, None]).sum(axis=1)

    rhs = t / (1 - ll)

    plt.plot(l, lhs)
    plt.plot(b, -m * b + c, 'o')

    plt.plot(ll, rhs)
    plt.plot(s, t / (1 - s), 's')

    for i in range(3):
        plt.plot(l, -m[i] * l + c[i])

    #plt.plot(l, rhs)
    plt.show()