import matplotlib as mpl
import matplotlib.pyplot as plt
from linadvtrain.classification import lin_advclasif, CostFunction
import numpy as np
import linadvtrain.cvxpy_impl as cvxpy_impl
import time
from datasets import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    X, y, X_test, y_test = MNIST()

    # Compute performance logistic regression
    clf = LogisticRegression(random_state=0, fit_intercept=False).fit(X, y)
    y_prob_lr = clf.predict_proba(X_test)[:, 1:]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    print(f'logistic={auc_lr}')

    # Compute performance adv-training
    params, info = lin_advclasif(X, y, verbose=True, p=np.inf, method='saga', max_iter=1000,
                                 batch_size=200)
    y_prob_adv = sigmoid(X_test @ params)
    auc_adv = roc_auc_score(y_test, y_prob_adv)
    print(f'advtrain={auc_adv}')