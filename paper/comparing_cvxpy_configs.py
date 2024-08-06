
import linadvtrain.cvxpy_impl as cvxpy_impl
import cvxpy as cp
from datasets import *
import time
from linadvtrain.regression import lin_advregr, get_radius

X_train_, X_test_, y_train, y_test = magic()
rng = np.random.RandomState(1)

df = pd.DataFrame({'#params': [], 'CVXPY': [], 'CVXPY (MOSEK)': [], 'OUR':[]})
for n_params in [30, 100, 300]:#, 1000, 3000, 10000, 30000]:
    print(n_params)
    indices = rng.choice(X_train_.shape[1], size=n_params)
    X_train = X_train_[:, indices]
    X_test = X_test_[:, indices]

    rng = np.random.RandomState(1)

    adv_radius = 0.001

    start_time = time.time()
    mdl = cvxpy_impl.AdversarialRegression(X_train, y_train, p=np.inf)

    start_time = time.time()
    params_cvxpy = mdl(adv_radius=adv_radius, verbose=False)
    t_normal = time.time() - start_time

    mdl = cvxpy_impl.AdversarialRegression(X_train, y_train, p=np.inf)
    start_time = time.time()
    params_cvxpy_mosek = mdl(adv_radius=adv_radius, verbose=False, solver=cp.MOSEK)
    t_mosek = time.time() - start_time

    start_time = time.time()
    params, info = lin_advregr(X_train, y_train, adv_radius=adv_radius, verbose=False, p=np.inf, method='w-ridge')
    t_ours =  time.time() - start_time

    df = df.append({'#params': n_params, 'CVXPY': t_normal , 'CVXPY (MOSEK)': t_mosek, 'OUR': t_ours}, ignore_index=True)
    df.to_csv('data/comparing_cvxpy.csv', index=False)

if __name__ == "__main__":
    print(df)