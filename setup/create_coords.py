import os
import sys

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)
from utils.utils import analyze_bounding_box, get_bounding_box


def main():

    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    META_DIRPATH = os.path.join(path, "..", "data", "metadata")
    COORD_FILEPATH= os.path.join(path, "..", "data", "labels", "coord.csv")

    with open(COORD_FILEPATH, 'w') as f:
        for meta_name in os.listdir(META_DIRPATH):
            m_path = os.path.join(META_DIRPATH, meta_name)
            bb = get_bounding_box(m_path)
            ul, ur, ll, lr, edges = analyze_bounding_box(bb)
            midpoint = [(ul[0] + lr[0]) / 2.0, (ul[1] + lr[1]) / 2.0]
            imagename = f"{meta_name[:-8]}"
            f.write(f"{imagename},{midpoint}\n")


if __name__ == "__main__":
    main()