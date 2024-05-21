# Adversarial training in linear models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This library solves linear regression models. It find the parameter $\hat \beta$ of a linear model, 
such that given an input $x$ and it produces the prediction.
$$\hat{y} = \hat\beta^\top x.$$


This library solves the regression problem (estimating the parameter $\hat\beta$) using adversarial training.
The idea is to make the model robust by optimizing in the presence of disturbances.
It considers training inputs have been contaminated with disturbances deliberately 
chosen to maximize the model error.

# Installation

Move into the downloaded directory and install requirements with:
```bash
pip install -r requirements.txt
```

In sequence, install package with:
```bash
python setup.py install
```

# Reproducing paper
You can reproduce the results and figures from the paper by running:
```sh
cd paper
bash generate_figs.sh
```