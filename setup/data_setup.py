import os
import sys


def main(args):

    path = os.path.abspath(__file__)
    path = os.path.dirname(path)

    LABELS_DIRPATH = os.path.join(path, "..", "data", "labels")

    # A list of steps to be performed
    BRIGHTNESS_FILE = "brightness.csv"
    LABELS_FILE = "labels_binary_minerals.csv"
    OCEAN_FILE = "local_ocean_mask.csv"
    COORD_FILE = "coords.csv"

    if args[0] == "--overwrite":
        print("Overwriting existing files ...")
        os.system(f"python3 {path}/compute_avg_brightness.py")
        os.system(f"python3 {path}/create-labels.py") 
        os.system(f"python3 {path}/mask_ocean_local.py")
        os.system(f"python3 {path}/create_coords.py")

    else:
        if not os.path.exists(os.path.join(LABELS_DIRPATH, BRIGHTNESS_FILE)):
            print("Computing average brightness ...")
            os.system(f"python3 {path}/compute_avg_brightness.py")
        
        if not os.path.exists(os.path.join(LABELS_DIRPATH, LABELS_FILE)):
            print("Creating labels ...")
            os.system(f"python3 {path}/create-labels.py") 

        if not os.path.exists(os.path.join(LABELS_DIRPATH, OCEAN_FILE)):
            print("Masking ocean ...")
            os.system(f"python3 {path}/mask_ocean_local.py")
        
        if not os.path.exists(os.path.join(LABELS_DIRPATH, COORD_FILE)):
            print("Creating coords ...")
            os.system(f"python3 {path}/create_coords.py")


if __name__ == "__main__":
    main(sys.argv)