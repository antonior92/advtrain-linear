# %% Imports
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import linear_model

#%% Define problem
X, y = datasets.load_breast_cancer(return_X_y=True)

X -= np.mean(X, axis=0)
X /= X.max(axis=0)  # Normalize each feature to be in [-1, 1]
y = np.asarray(y, dtype=np.float64)

#%% Implement gradiend descent in logistic regression
C = 1e-3
clf = linear_model.LogisticRegression(penalty='l2', C=C, fit_intercept=False)
clf.fit(X, y)
clf.coef_


# Implement gradient descent
n_iter = 1000
dist = np.zeros(n_iter)
lr = 0.1
sigmoid = lambda x: 1 / (1 + np.exp(-x))
# convert y to +1 or - 1
V = (2 * y - 1)[:, None] * X
w = np.zeros(V.shape[1])
for i in range(n_iter):
    update = - C * V.T @ (1-sigmoid(V @ w)) + w
    w = w - lr * update
    print(np.linalg.norm(update))
    dist[i] = np.linalg.norm(w - clf.coef_.ravel())

#%% plot distance
import matplotlib.pyplot as plt

plt.plot(dist)
plt.yscale('log')
plt.show()

if __name__ == "__main__":
    pass