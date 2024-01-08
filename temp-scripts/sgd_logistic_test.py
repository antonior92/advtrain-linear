# %% Imports
import numpy as np
from sklearn import linear_model
from linadvtrain.first_order_methods import gd, sgd

sigmoid = lambda x: 1 / (1 + np.exp(-x))

#%% Define dataset
n_params = 500
n_train = 1000
rng = np.random.RandomState(1)
beta = 1 / np.sqrt(n_params) * np.ones(n_params)
X = rng.randn(n_train, n_params)

# Generate output with random additive noise
e = rng.randn(n_train)
y = np.sign(X @ beta + e)

# Define logistic regression
C = 1e-3

class LogisticReg(object):
    def __init__(self, X, y, C):
        self.X = X
        self.y = y
        self.V = y[:, None] * X
        self.C = C

    def grad(self, w, i=None):
        if i is None:
            Vi = self.V
        elif isinstance(i, int) or isinstance(i, np.int64):
            Vi = self.V[i:i+1, :]
        else:
            raise ValueError

        grad = - C * Vi.T @ (1 - sigmoid(Vi @ w)) + w * Vi.shape[0] / n_train
        return grad

logreg = LogisticReg(X, y, C)

#%%  Compute problem matrix
clf = linear_model.LogisticRegression(penalty='l2', C=C, fit_intercept=False)
clf.fit(X, y)
clf.coef_

#%% Implement gradiend descent in logistic regression
n_iter = 40
lr = 0.1

w0 = np.zeros(n_params)

dist_gd = np.zeros(n_iter)
def callback_gd(i, _new_w, w):
    dist_gd[i] = np.linalg.norm(w - clf.coef_.ravel())

w = gd(w0, logreg.grad, lr=lr, max_iter=n_iter, callback=callback, momentum=0.1)


#%% Implement stochastic gradiend descent in logistic regression
dist_sgd = np.zeros(n_iter)
w = np.zeros(n_params)
lr = 1
for i in range(n_iter):
    dist_sgd[i] = np.linalg.norm(w - clf.coef_.ravel())
    indexes = np.random.permutation(np.arange(n_train))
    for s in indexes:
        update = logreg.grad(w, s)
        w = w - lr * update

#%% Implement SAGA in logistic regression
dist_saga = np.zeros(n_iter)
gradient_buffer = np.zeros_like(X)
sum_grads = np.zeros_like(n_params)
w = np.zeros(n_params)
lr = 1
for i in range(n_iter):
    dist_saga[i] = np.linalg.norm(w - clf.coef_.ravel())
    indexes = np.random.permutation(np.arange(n_train))
    for s in indexes:
        grad = logreg.grad(w, s)
        grad_old = gradient_buffer[s, :].copy()
        gradient_buffer[s, :] = grad
        update = grad - grad_old + sum_grads
        sum_grads = sum_grads + 1 / n_train * (grad - grad_old)
        w = w - lr * update

#%% Plot all
import matplotlib.pyplot as plt

plt.plot(dist_sgd, label='sgd')
plt.plot(dist_gd, label='gd')
plt.plot(dist_saga, label='saga', color='red')
plt.legend()
plt.yscale('log')
plt.show()

#%% main
if __name__ == "__main__":
    pass