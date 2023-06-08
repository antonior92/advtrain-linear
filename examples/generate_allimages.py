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
    for adv_radius in 0.001*np.arange(1, 50):
        one_dimensional(adv_radius)
        img_path = src_path + f'/one_dimensional/one_dimensional_{adv_radius}.png'
        plt.savefig(img_path)
        frames.append(Image.open(img_path))

    # Save gif
    frames[0].save(src_path + '/one_dimensional.gif', format='GIF',
                   append_images=frames[1:], save_all=True, duration=200,
                   loop=0)
