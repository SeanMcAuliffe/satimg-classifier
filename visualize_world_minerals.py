import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

minerals_dir_path = os.path.join("data", "minerals")

def load_minerals(filename):
    # Assumes csv files are in directory /data/minerals/
    # "minerals" is a list of lists, where each sublist is a row of the csv.
    minerals_file_path = os.path.join(minerals_dir_path, filename)
    minerals_rows = pd.read_csv(minerals_file_path, header=0, low_memory=False)
    minerals_rows = minerals_rows.replace({np.nan:None})
    minerals = minerals_rows.values.tolist()

    # In minerals.csv, some deposits lack lat-lon coords. Remove these.
    minerals = [record for record in minerals if (record[1] and record[2])]

    return minerals
minerals = load_minerals("minerals.csv")
minerals_cleaned = load_minerals("minerals_cleaned.csv")

# Swap coords to be (longitude, latitude) instead of (latitude, longitude),
# so that they can be plotted on a world map.
deposit_coords_m = [(record[2], record[1]) for record in minerals]
deposit_coords_mc = [(record[2], record[1]) for record in minerals_cleaned]

# Plot the positive and negative points on a world map.
def plot_on_world_map(deposits, title, xlabel, ylabel):
    bg_image = mpimg.imread("world-map.png")

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor((1, 1, 1, 0))

    x_vals = [point[0] for point in deposits]
    y_vals = [point[1] for point in deposits]
    ax.plot(x_vals, y_vals, marker=".", markersize=0.5, lw=0.01, linestyle="None", color="blue", label="pos")

    ax.imshow(bg_image, extent=[-180.0, 180.0, -90.0, 90.0])

    plt.xlim([-180.0, 180.0])
    plt.ylim([-90.0, 90.0])

plot_on_world_map(deposit_coords_m, title="Minerals", xlabel="longitude", ylabel="latitude")
plt.savefig("world-map-minerals.png", dpi=1200)
plot_on_world_map(deposit_coords_mc, title="Minerals (Cleaned)", xlabel="Longitude", ylabel="Latitude")
plt.savefig("world-map-minerals-cleaned.png", dpi=1200)
plt.show()
plt.close()

