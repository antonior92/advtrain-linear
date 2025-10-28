import numpy as np
from linadvtrain.datasets import load_magic
import sklearn.model_selection
from sklearn.linear_model import ElasticNetCV

if __name__ == '__main__':
    from linadvtrain.regression import lin_advregr, get_radius

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
    nmse_lasso = np.mean((y_test - y_pred) ** 2) / np.mean(y_test ** 2)
    print(nmse_lasso)

    # Adversarial estimation
    estimated_params, info = lin_advregr(X_train, y_train, p=np.inf)
    y_pred = X_test @ estimated_params
    nmse_advlinf = np.mean((y_test - y_pred) ** 2) / np.mean(y_test ** 2)
    print(nmse_advlinf)


