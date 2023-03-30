import numpy as np
import subprocess
import os


def resize_512(image: np.ndarray):
    """Provided an image of at most 512x512 pixels,
    resize it to exactly 512x512"""

    shape = image.shape

    x = shape[0]
    y = shape[1]

    image = np.pad(image, ((0, 512-x), (0, 512-y)), mode='constant', constant_values=0)

    return image