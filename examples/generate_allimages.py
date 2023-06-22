from one_dimensional import one_dimensional
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

if __name__ == '__main__':
    src_path = '../imgs'
    # Generate one-dimensional image
    os.makedirs(src_path, exist_ok=True)
    frames = []
    one_dimensional()
    plt.savefig(src_path + f'/one_dimensional.png')
