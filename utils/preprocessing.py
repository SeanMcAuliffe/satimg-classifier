import numpy as np
import os
from scipy.ndimage import zoom
import tifffile

def resize_512(image: np.ndarray):
    """ Provided an image of at most 512x512 pixels,
        resize it to exactly 512x512 """
    shape = image.shape
    x = shape[0]
    y = shape[1]
    image = np.pad(image,
                   ((0, 512-x),
                    (0, 512-y)),
                    mode='constant',
                    constant_values=0)
    return image


def downsample_to(image: np.ndarray, size: int) -> np.ndarray:
    """ Downscale the image to (size x size) pixels"""
    zoom_amount = size / image.shape[0]
    return zoom(image, zoom_amount)
