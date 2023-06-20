import numpy as np
from linadvtrain.datasets import load_magic
import sklearn.model_selection
from sklearn.linear_model import ElasticNetCV

if __name__ == '__main__':
    from linadvtrain.solvers import lin_advtrain

    X, y = load_magic(input_folder='../WEBSITE/DATA')

    # Train-test split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50,
                                                                                random_state=0)

    # Rescale
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Compute lasso performance
    print("elastic net...")
    regr = ElasticNetCV(cv=10, random_state=0, l1_ratio=1)
    regr.fit(X_train, y_train)
    y_pred = X_test @ regr.coef_
    nmse = np.mean((y_test - y_pred) ** 2) / np.mean(y_test ** 2)

    # Adversarial estimation
    adv_radius_0 = np.linalg.norm(X.T @ y, ord=np.inf) / np.abs(y).sum()
    e = np.random.randn(X.shape[0], 100)
    adv_radius_est = np.mean(np.linalg.norm(X.T @ e, ord=np.inf, axis=0) / np.sum(np.abs(e), axis=0))
    adv_radius = np.linalg.norm(X, ord=2, axis=0).mean() * np.sqrt(np.pi * np.log(2 * X.shape[1])) / X.shape[0]
    estimated_params, info = lin_advtrain(X_train, y_train, adv_radius=adv_radius_est, max_iter=100, p=np.inf, verbose=True, method='w-ridge')
    estimated_params, info = lin_advtrain(X_train, y_train, adv_radius=adv_radius, max_iter=100, p=np.inf,
                                          verbose=True, method='w-ridge')

    y_pred = X_test @ estimated_params
    nmse = np.mean((y_test - y_pred) ** 2) / np.mean(y_test ** 2)
    print(nmse)


