# Adversarial training in linear models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This library solves linear regression models. It find the parameter $\hat \beta$ of a linear model, 
such that given an input $x$ and it produces the prediction.
$$\hat{y} = \hat\beta^\top x.$$


This library solves the regression problem (estimating the parameter $\hat\beta$) using adversarial training.
The idea is to make the model robust by optimizing in the presence of disturbances.
It considers training inputs have been contaminated with disturbances deliberately 
chosen to maximize the model error.

Given pair of input-output samples $(x_i, y_i), i = 1, \dots, n$, it is formulated as a min-max optimization problem:

$$\min_\beta \frac{1}{n} \sum_{i=1}^n \max_{\|\Delta x_i\| \le \delta} (y_i - \beta^\top(x_i+ \Delta x_i))^2$$

## Usage

```python
from linadvtrain.solvers import lin_advtrain
import numpy as np

# Generate dataset
rng = np.random.RandomState(5)
X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])[:, None]
noise = 0.15 * rng.randn(len(x))
y = x + noise

# Adversarial estimation
adv_radius = 0.05
estimated_params, info = lin_advtrain(X, y, adv_radius=adv_radius)
```

The gif below shows the adversarial training the output of the example above, with the adversarial radius highlighted.
See [examples/one_dimensional.py](examples/one_dimensional.py) for more details.

![one](imgs/one_dimensional.png)



## Installation

## Benefits of adversarial training