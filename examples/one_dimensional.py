from linadvtrain.solvers import lin_advtrain, get_radius
import numpy as np
import matplotlib.pyplot as plt


def one_dimensional(adv_radius=0.05):
    # Generate dataset
    rng = np.random.RandomState(5)
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    noise = 0.1 * rng.randn(len(x))
    y = x + noise

    # Adversarial estimation
    estimated_params, info = lin_advtrain(x[:, None], y, adv_radius=adv_radius)

    # Plot dataset
    x_probe = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=adv_radius, ls='',
                 marker='o', color='b', capsize=3.5,
                 label='data points + adv. region')
    ax.plot(x_probe, estimated_params * x_probe, 'k-', label='estimated')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.legend()
    plt.grid()


if __name__ == '__main__':
    one_dimensional()
    plt.show()
