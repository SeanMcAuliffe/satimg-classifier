import numpy as np
import subprocess
import os
import tifffile

def resize_512(image: np.ndarray):
    """Provided an image of at most 512x512 pixels,
    resize it to exactly 512x512"""

    shape = image.shape
    
    x = shape[0]
    y = shape[1]

    image = np.pad(image, (512-x, 512-y), mode='constant', constant_values=(0))

    return image