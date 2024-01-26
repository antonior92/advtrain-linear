import sklearn.model_selection
from sklearn.datasets import load_diabetes
from ucimlrepo import fetch_ucirepo, list_available_datasets
import numpy as np
import pandas as pd
import os

def load_magic(input_folder='WEBSITE/DATA', output_phenotype='HET_2'):
    # You can download and extract the MAGIC dataset by
    # > wget http://mtweb.cs.ucl.ac.uk/mus/www/MAGICdiverse/MAGIC_diverse_FILES/BASIC_GWAS.tar.gz
    # > tar -xvf BASIC_GWAS.tar.gz
    # Load data
    founder_names = ["Banco", "Bersee", "Brigadier", "Copain", "Cordiale", "Flamingo",
                     "Gladiator", "Holdfast", "Kloka", "MarisFundin", "Robigus", "Slejpner",
                     "Soissons", "Spark", "Steadfast", "Stetson"]

    # Genotype
    genotype = pd.read_csv(os.path.join(input_folder, 'MAGIC_IMPUTED_PRUNED/MAGIC_imputed.pruned.traw'),
                           sep='\t')
    genotype.set_index('SNP', inplace=True)
    genotype = genotype.iloc[:, 5:]
    colnames = genotype.keys()
    new_colnames = [c.split('_')[0] for c in colnames]
    genotype.rename(columns={c: new_c for c, new_c in zip(colnames, new_colnames)}, inplace=True)
    genotype = genotype.transpose()

    # Phenotype
    phenotype = pd.read_csv(os.path.join(input_folder, 'PHENOTYPES/NDM_phenotypes.tsv'), sep='\t')
    phenotype.set_index('line_name', inplace=True)
    phenotype.drop(founder_names, inplace=True)
    del phenotype['line_code']

    # Make genotype have the same index as phenotype
    genotype = genotype.reindex(phenotype.index, )

    # Replace NAs
    genotype = genotype.fillna(genotype.mean(axis=0))
    phenotype = phenotype.fillna(phenotype.mean(axis=0))

    # Formulate problem
    X = np.array(genotype.values)
    y = np.array(phenotype[output_phenotype].values)

    return X, y

def normalize(X_train, X_test, y_train, y_test):
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return X_train, X_test, y_train, y_test

# DATASET REGRESSION
def magic():
    X, y = load_magic(input_folder='../WEBSITE/DATA')
    # Train-test split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50, random_state=0)
    X_train = X_train
    X_test = X_test
    return normalize(X_train, X_test, y_train, y_test)

def diabetes():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50, random_state=0)
    return normalize(X_train, X_test, y_train, y_test)

def wine():
    dset = fetch_ucirepo(name="Wine Quality")
    X = dset.data.features.values
    y = dset.data.targets.values.flatten()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    return normalize(X_train, X_test, y_train, y_test)

def abalone():
    dset = fetch_ucirepo(name="Abalone")
    F = dset.data.features
    F = F.assign(Sex=(F['Sex'] == 'M').values.astype(float))
    X = F.values
    y = dset.data.targets.values.flatten()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    return normalize(X_train, X_test, y_train, y_test)

def heart_failure():
    dset = fetch_ucirepo(name="Heart failure clinical records")
    F = dset.data.features
    X = F.values
    y = dset.data.targets.values.flatten()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    return normalize(X_train, X_test, y_train, y_test)
