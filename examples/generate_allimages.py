from one_dimensional import one_dimensional
from diabetes_path import diabetes_path
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

if __name__ == '__main__':
    src_path = '../imgs'
    os.makedirs(src_path, exist_ok=True)

    # Generate one-dimensional image
    one_dimensional()
    plt.savefig(src_path + f'/one_dimensional.png')

    # Generate image for diabetes dataset
    diabetes_path()
    for lbl in plt.get_figlabels():
        plt.figure(lbl)
        plt.savefig(src_path +'/' + lbl + '.png')
