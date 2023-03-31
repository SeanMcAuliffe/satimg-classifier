# Generic Utility / Helper Functions 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os 


VIS_DIR_PATH = os.path.join("..", "vis")


def get_bounding_box(md_filepath):
    # From metadata file, obtain coords of bounding box of image.
    bounding_box = {}
    with open(md_filepath) as md_file:
        # Scan lines only until all corner coords are found.
        for line in md_file:
            key, val  = line.partition("=")[::2]
            key = key.strip()
            val = val.strip()
            if key.endswith("LAT_PRODUCT") or key.endswith("LON_PRODUCT"):
                bounding_box[key] = float(val)
            if len(bounding_box) == 8:
                break
    return bounding_box


def analyze_bounding_box(bounding_box):
    # Obtain vectors of bounding box edges.
    ul = [bounding_box["CORNER_UL_LAT_PRODUCT"], bounding_box["CORNER_UL_LON_PRODUCT"]]
    ur = [bounding_box["CORNER_UR_LAT_PRODUCT"], bounding_box["CORNER_UR_LON_PRODUCT"]]
    ll = [bounding_box["CORNER_LL_LAT_PRODUCT"], bounding_box["CORNER_LL_LON_PRODUCT"]]
    lr = [bounding_box["CORNER_LR_LAT_PRODUCT"], bounding_box["CORNER_LR_LON_PRODUCT"]]
    edges = [] # Has form [edge start point, edge direction]
    edges.append([ul, [(ur[0] - ul[0]), (ur[1] - ul[1])]])
    edges.append([ur, [(lr[0] - ur[0]), (lr[1] - ur[1])]])
    edges.append([lr, [(ll[0] - lr[0]), (ll[1] - lr[1])]])
    edges.append([ll, [(ul[0] - ll[0]), (ul[1] - ll[1])]])
    return ul, ur, ll, lr, edges
    

def plot_on_world_map(pos, neg, title, xlabel, ylabel, filename):
    bg_image = mpimg.imread(os.path.join(VIS_DIR_PATH, "world-map.png"))

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor((1, 1, 1, 0))

    x_vals = [point[0] for point in pos]
    y_vals = [point[1] for point in pos]
    ax.plot(x_vals, y_vals, marker=".", markersize=0.5, lw=0.01, linestyle="None", color="blue", label="pos")

    x_vals = [point[0] for point in neg]
    y_vals = [point[1] for point in neg]
    ax.plot(x_vals, y_vals, marker=".", markersize=0.5, lw=0.01, linestyle="None", color="red", label="neg")

    ax.imshow(bg_image, extent=[-180.0, 180.0, -90.0, 90.0])

    plt.xlim([-180.0, 180.0])
    plt.ylim([-90.0, 90.0])
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(VIS_DIR_PATH, filename), dpi=1200)