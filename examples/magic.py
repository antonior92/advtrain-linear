
import numpy as np
import pandas as pd
import os
import sklearn
import sklearn.model_selection
from linadvtrain.datasets import load_magic

if __name__ == '__main__':
    from linadvtrain.solvers import lin_advtrain

    X, y = load_magic(input_folder='../WEBSITE/DATA')
    X -= X.mean(axis=0)
    y -= y.mean()
    # Adversarial estimation
    estimated_params, info = lin_advtrain(X, y, adv_radius=0.001, max_iter=100, p=np.inf, verbose=True, method='w-ridge')
