from PIL import Image
import numpy as np
import tifffile
import os

from utils.preprocessing import resize_512


image_path = "./data/images/"
norm_path = "./data/images_norm/"

if not os.path.exists(norm_path):
    os.mkdir(norm_path)
    

def main():
    print("Starting...")
    for imagename in os.listdir(image_path):
        im = np.array(Image.open(os.path.join(image_path, imagename)))
        im = resize_512(im)
        tifffile.imwrite(os.path.join(norm_path, imagename), im)


if __name__ == "__main__":
    main()
