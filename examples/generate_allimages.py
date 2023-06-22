from one_dimensional import one_dimensional
from diabetes_path import diabetes_path
from transition_into_interpolation import transition_into_interpolation
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    src_path = '../imgs'
    os.makedirs(src_path, exist_ok=True)

    # Generate one-dimensional image
    one_dimensional()
    plt.savefig(src_path + f'/one_dimensional.png')

    # Generate image for diabetes dataset
    diabetes_path()
    path = src_path + '/diabetes'
    os.makedirs(path, exist_ok=True)
    for lbl in plt.get_figlabels():
        plt.figure(lbl)
        plt.savefig(path +'/' + lbl + '.png')

    # Generate transition into interpolation image
    transition_into_interpolation()
    path = src_path + '/transition_into_interpolation'
    os.makedirs(path, exist_ok=True)
    for lbl in plt.get_figlabels():
        plt.figure(lbl)
        plt.savefig(path +'/' + lbl + '.png')