import numpy as np
import os
from PIL import Image
import tifffile


image_path = "./data/images/"
norm_path = "./data/images_norm/"


def resize_512(image: np.ndarray):
    """Provided an image of at most 512x512 pixels,
    resize it to exactly 512x512"""
    shape = image.shape
    x = shape[0]
    y = shape[1]
    image = np.pad(image, ((0, 512-x), (0, 512-y)), mode='constant', constant_values=0)
    return image


def main():
    print("Starting...")
    for imagename in os.listdir(image_path):
        im = np.array(Image.open(os.path.join(image_path, imagename)))
        im = resize_512(im)
        tifffile.imsave(os.path.join(norm_path, imagename), im)


if __name__ == "__main__":
    main()
