import os
import geopandas as gpd
from shapely.geometry import Point
import sys


parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)
from utils.utils import get_bounding_box, analyze_bounding_box, plot_on_world_map_pos_neg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


#IMG_DIRPATH = os.path.join(".", "data", "images_norm")
META_DIRPATH = os.path.join("..", "data", "metadata")
OUT_PATH = os.path.join("..", "data", "labels", "ocean_mask.csv")
NE_10M_PATH = os.path.join("..", "ne_110m_land", "ne_110m_land.shp")
VIS_DIR_PATH = os.path.join("..", "vis")


def mask_ocean():

    if os.path.exists(OUT_PATH):
        return

    world = gpd.read_file(NE_10M_PATH)
    coords = {}
    
    for meta_name in os.listdir(META_DIRPATH):
        m_path = os.path.join(META_DIRPATH, meta_name)
        bb = get_bounding_box(m_path)
        ul, ur, ll, lr, edges = analyze_bounding_box(bb)
        midpoint = [(ul[0] + lr[0]) / 2.0, (ul[1] + lr[1]) / 2.0]
        coords[meta_name[:-8]] = midpoint

    def is_over_land(latitude, longitude):
        point = Point(longitude, latitude)
        for _, country in world.iterrows():
            if country['geometry'].contains(point):
                return True
        return False

    print("Starting to remove the world's oceans...")
    with open(OUT_PATH, "w") as f:
        for key in coords.keys():
            lat, lng = coords[key]
            on_land = is_over_land(lat, lng)
            #print(f"{key} @ ({lat}, {long} on land: {on_land})")
            if on_land:
                f.write(f"{key},1\n")
            else:
                f.write(f"{key},0\n")


def plot_results():

    coords = {}
    for meta_name in os.listdir(META_DIRPATH):
        m_path = os.path.join(META_DIRPATH, meta_name)
        bb = get_bounding_box(m_path)
        ul, ur, ll, lr, edges = analyze_bounding_box(bb)
        midpoint = [(ul[0] + lr[0]) / 2.0, (ul[1] + lr[1]) / 2.0]
        coords[meta_name[:-8]] = midpoint

    labels = {}
    with open(OUT_PATH, 'r') as f:
        for line in f.readlines():
            name, label = line.rstrip().split(',')
            labels[name] = int(label)

    points_to_plot = []
    for key in coords.keys():
        if labels[key] == 1:
            points_to_plot.append(coords[key])

    points_to_plot = [(p[1], p[0]) for p in points_to_plot]

    bg_image = mpimg.imread(os.path.join(VIS_DIR_PATH, "world-map.png"))
    
    fig, ax = plt.subplots()
    ax.set_title("hopefully no ocean")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_facecolor((1, 1, 1, 0))

    x_vals = [point[0] for point in points_to_plot]
    y_vals = [point[1] for point in points_to_plot]
    ax.plot(x_vals, y_vals, marker=".", markersize=0.5, lw=0.01, linestyle="None", color="blue", label="pos")

    ax.imshow(bg_image, extent=[-180.0, 180.0, -90.0, 90.0])

    plt.xlim([-180.0, 180.0])
    plt.ylim([-90.0, 90.0])
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(VIS_DIR_PATH, "with-ocean-mask.png"), dpi=1200)

    

if __name__ == "__main__":
    mask_ocean()
    plot_results()
