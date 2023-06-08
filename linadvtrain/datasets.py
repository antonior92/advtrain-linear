import numpy as np
import pandas as pd
import os
import sklearn.model_selection


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
