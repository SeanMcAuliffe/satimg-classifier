# Visualize examples as points on a world map.
# By default, plots all examples.
# If a batch name is provided as command line arg,
# plots only examples from that batch.

import sys
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# Initialize paths.
metadata_dir_path = os.path.join("data", "metadata")
labels_dir_path = os.path.join("data", "labels")
vis_dir_path = os.path.join("vis")

# Get labels CSV filename from command line.
if len(sys.argv) > 2:
    print("Usage: python visualize_world.py <opt: labels_batch>")
    sys.exit(1)

user_batch_name = None
if len(sys.argv) == 2:
    user_batch_name = sys.argv[1]
    if user_batch_name not in os.listdir(labels_dir_path):
        print("Error: batch name not found in labels directory.")
        sys.exit(1)
    if user_batch_name not in os.listdir(metadata_dir_path):
        print("Error: batch name not found in metadata directory.")
        sys.exit(1)

# Load labels.
labels_bm = []
labels_bmc = []

# for batch_name in os.listdir(labels_dir_path):
#     if user_batch_name is not None and batch_name != user_batch_name:
#         continue

batch_path = metadata_dir_path # os.path.join(metadata_dir_path)
labels_batch_path = labels_dir_path # os.path.join(labels_dir_path)

labels_bm_file_path = os.path.join(labels_batch_path, "labels_binary_minerals.csv")
labels_bmc_file_path = os.path.join(labels_batch_path, "labels_binary_minerals_cleaned.csv")

labels_bm_rows = pd.read_csv(labels_bm_file_path, header=0, low_memory=False)
labels_bmc_rows = pd.read_csv(labels_bmc_file_path, header=0, low_memory=False)

labels_bm += labels_bm_rows.values.tolist()
labels_bmc += labels_bmc_rows.values.tolist()

labels_bm = dict(labels_bm)
labels_bmc = dict(labels_bmc)

# Create a lists of the coords of positive and negative examples.
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

latlon_pos_bm = []
latlon_neg_bm = []
latlon_pos_bmc = []
latlon_neg_bmc = []

# For each example. . .
# for batch_name in os.listdir(metadata_dir_path):
#     if user_batch_name is not None and batch_name != user_batch_name:
#         continue

batch_path = metadata_dir_path # os.path.join(metadata_dir_path, batch_name)
for filename in os.listdir(batch_path):
    metadata_file_path = os.path.join(batch_path, filename)

    # Obtain centerpoint of image.
    # Obtain label.
    # Add to lists by label and axis, for plotting.
    bb = get_bounding_box(metadata_file_path)
    ul, ur, ll, lr, edges = analyze_bounding_box(bb)
    midpoint = [(ul[0] + lr[0]) / 2.0, (ul[1] + lr[1]) / 2.0]
    imagename = os.path.splitext(filename)[0]
    label_bm = labels_bm[imagename]
    label_bmc = labels_bmc[imagename]

    if label_bm == 1:
        latlon_pos_bm.append(midpoint)
    else:
        latlon_neg_bm.append(midpoint)
    if label_bmc == 1:
        latlon_pos_bmc.append(midpoint)
    else:
        latlon_neg_bmc.append(midpoint)

# Plot the positive and negative points on a world map.
def plot_on_world_map(pos, neg, title, xlabel, ylabel):
    bg_image = mpimg.imread(os.path.join(vis_dir_path, "world-map.png"))

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

# Swap coordinates to match world map.
lonlat_pos_bm = [[point[1], point[0]] for point in latlon_pos_bm]
lonlat_neg_bm = [[point[1], point[0]] for point in latlon_neg_bm]
lonlat_pos_bmc = [[point[1], point[0]] for point in latlon_pos_bmc]
lonlat_neg_bmc = [[point[1], point[0]] for point in latlon_neg_bmc]

# Plot on cartesian axes.
title1 = "pos and neg examples, minerals.csv"
title1 += ", " + user_batch_name if user_batch_name is not None else ""
filename1 = "world-map-examples-minerals"
filename1 += "-" + user_batch_name if user_batch_name is not None else ""
plot_on_world_map(lonlat_pos_bm, lonlat_neg_bm, title=title1, xlabel="longitude", ylabel="latitude")
plt.savefig(os.path.join(vis_dir_path, filename1), dpi=1200)
title2 = "pos and neg examples, minerals_cleaned.csv"
title2 += ", " + user_batch_name if user_batch_name is not None else ""
filename2 = "world-map-examples-minerals-cleaned"
filename2 += "-" + user_batch_name if user_batch_name is not None else ""
plot_on_world_map(lonlat_pos_bmc, lonlat_neg_bmc, title=title2, xlabel="longitude", ylabel="latitude")
plt.savefig(os.path.join(vis_dir_path, filename2), dpi=1200)
plt.show()
plt.close()

