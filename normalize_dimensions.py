# 
from PIL import Image
import numpy as np
import tifffile
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from utils.preprocessing import resize_512


image_path = "./data/images/"
norm_path = "./data/images_norm/"


def main():
    print("Starting...")
    for imagename in os.listdir(image_path):
        im = np.array(Image.open(os.path.join(image_path, imagename)))
        im = resize_512(im)
        tifffile.imsave(os.path.join(norm_path, imagename), im)


if __name__ == "__main__":
    main()
