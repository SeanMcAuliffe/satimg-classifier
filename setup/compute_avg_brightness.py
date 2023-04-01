from PIL import Image
import numpy as np
import os    

path = os.path.abspath(__file__)
path = os.path.dirname(path)
IMG_DIRPATH = os.path.join(path, "..", "data", "images_norm")
META_DIRPATH = os.path.join(path, "..", "data", "metadata")
BRIGHT_FILEPATH= os.path.join(path, "..", "data", "labels", "brightness.csv")


def main():
    with open(BRIGHT_FILEPATH, 'w') as f:
        for img_name in os.listdir(IMG_DIRPATH):
            img = np.array(Image.open(os.path.join(IMG_DIRPATH, f"{img_name}")))
            m = np.mean(img)
            f.write(f"{img_name[:-7]},{m}\n")


if __name__ == "__main__":
    main()