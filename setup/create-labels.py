# Create labels for landsat images.

import pandas as pd
import numpy as np
import os.path
import math
import time
from sklearn.neighbors import KDTree

################################################################
# Configuration.
################################################################

minerals_dir_path = os.path.join("..", "data", "minerals")
metadata_dir_path = os.path.join("..", "data", "metadata")
labels_dir_path = os.path.join("..", "data", "labels")

################################################################
# Functions.
################################################################

def getRecordIndex(r):
    return r[0]

def get_kd_tree(minerals):
    coords = [([record[1], record[2]]) for record in minerals]
    tree = KDTree(np.array(coords))
    return tree

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

def load_all_minerals_and_kd_trees():
    minerals = load_minerals("minerals.csv")
    minerals_cleaned = load_minerals("minerals_cleaned.csv")
    minerals_kd_tree = get_kd_tree(minerals)
    minerals_cleaned_kd_tree = get_kd_tree(minerals_cleaned)

    return minerals, minerals_cleaned, minerals_kd_tree, minerals_cleaned_kd_tree

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

def get_deposit_indices_approx(ul, ur, ll, lr, kd_tree):
    # Use kd tree to obtain set of indices of deposits within a circle enclosing the bounding box.
    midpoint = [(ul[0] + lr[0]) / 2.0, (ul[1] + lr[1]) / 2.0]
    distance_ul_lr = math.sqrt(math.pow((ul[0] - lr[0]), 2) + math.pow((ul[1] - lr[1]), 2))
    distance_ur_ll = math.sqrt(math.pow((ur[0] - ll[0]), 2) + math.pow((ur[1] - ll[1]), 2))
    distance = max(distance_ul_lr, distance_ur_ll)
    target = np.array([midpoint])
    radius = distance* 0.5
    return kd_tree.query_radius(target, r=radius)[0]

def get_deposit_records(bounding_box, minerals, kd_tree=None):
    # Using image bounding box, find all mineral deposits present in the image.
    # Returns list of lists each of form [index, record].

    # Obtain bounding box corners and edges.
    ul, ur, ll, lr, edges = analyze_bounding_box(bounding_box)

    # If k-d tree provided, use it to efficiently first find deposits roughly in bounding box.
    candidates = get_deposit_indices_approx(ul, ur, ll, lr, kd_tree) if kd_tree else list(range(0, len(minerals)))

    # Of the candidate deposits, find those within bounding box.
    deposit_records = [] # Has form [index, record]
    for index in candidates:
        deposit_record = minerals[index]
        deposit_coords = [deposit_record[1], deposit_record[2]]
        # Image is a convex polygon. Then if deposit is on right-hand side of each edge, deposit is in the image.
        deposit_in_img = True
        for edge in edges:
            edge_start = edge[0]
            edge_dir = edge[1]
            edge_normal = [edge_dir[1], edge_dir[0] * -1.0]
            to_deposit = [deposit_coords[0] - edge_start[0], deposit_coords[1] - edge_start[1]]
            dot_prod = (edge_normal[0] * to_deposit[0]) + (edge_normal[1] * to_deposit[1])
            if dot_prod > 0:
                deposit_in_img = False
                break
        if deposit_in_img:
            deposit_records.append([index, deposit_record])
    return deposit_records

def convert_to_multiclass_label(deposit_records):
    # TODO: convert a list of deposit records into a multiclass label indicating which minerals are present.
    return

def generate_label(md_filepath, minerals, kd_tree, label_type="binary"):
    # From metadata filename and minerals dataset, create label for image.
    bounding_box = get_bounding_box(md_filepath)
    deposit_records = get_deposit_records(bounding_box, minerals, kd_tree)

    label = []
    if label_type == "binary":
        label = [0] if len(deposit_records) == 0 else [1]
    elif label_type == "multiclass":
        label = deposit_records
        # label = convert_to_multiclass_label(deposit_records)
        
    return label

def generate_and_store_all_labels():
    # Obtain metadata for all images of all batches.
    # Compute labels for all obtained imageds.
    # Store labels per batch in csv files having rows of form (imagename, label).

    # Load minerals dataset.
    print("Loading minerals dataset. . .")
    minerals, minerals_cleaned, minerals_kd_tree, minerals_cleaned_kd_tree = load_all_minerals_and_kd_trees()
    print("Loaded " + str(len(minerals)) + " rows from minerals.csv.")
    print("Loaded " + str(len(minerals_cleaned)) + " rows from minerals_cleaned.csv.")

    labels_bm_file_path = os.path.join(labels_dir_path, "labels_binary_minerals.csv")
    labels_bmc_file_path = os.path.join(labels_dir_path, "labels_binary_minerals_cleaned.csv")

    # For each metadata file. . .
    pairs_bm = []
    pairs_bmc = []
    for filename in os.listdir(metadata_dir_path):
        # Generate binary labels.
        metadata_file_path = os.path.join(metadata_dir_path, filename)
        label_bm = generate_label(metadata_file_path, minerals, minerals_kd_tree, label_type="binary")
        label_bmc = generate_label(metadata_file_path, minerals_cleaned, minerals_cleaned_kd_tree, label_type="binary")

        # Generate pairs of form (imagename, label).
        image_name = os.path.splitext(filename)[0]
        pairs_bm.append([image_name, label_bm[0]])
        pairs_bmc.append([image_name, label_bmc[0]])

    # Store labels.
    print("Storing labels. . .")
    pairs_bm = pd.DataFrame(pairs_bm)
    pairs_bmc = pd.DataFrame(pairs_bmc)
    if not os.path.exists(labels_dir_path):
        os.makedirs(labels_dir_path)
    pairs_bm.to_csv(labels_bm_file_path, index=False, header=["imagename", "label"])
    pairs_bmc.to_csv(labels_bmc_file_path, index=False, header=["imagename", "label"])
    
    print("Done.")
    return

################################################################
# Tests.
################################################################

def test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals, kd_tree, print_countries=False):
    # For a square region with sides aligned with lat/lon axes, verify result against a naive method.

    # Create dummy, square bounding box:
    bounding_box = {}
    bounding_box["CORNER_UL_LAT_PRODUCT"] = lat_north
    bounding_box["CORNER_UL_LON_PRODUCT"] = lon_west
    bounding_box["CORNER_UR_LAT_PRODUCT"] = lat_north
    bounding_box["CORNER_UR_LON_PRODUCT"] = lon_east
    bounding_box["CORNER_LL_LAT_PRODUCT"] = lat_south
    bounding_box["CORNER_LL_LON_PRODUCT"] = lon_west
    bounding_box["CORNER_LR_LAT_PRODUCT"] = lat_south
    bounding_box["CORNER_LR_LON_PRODUCT"] = lon_east

    # Use our algorithm, with and without kd_tree to find mineral deposits in bounding box.
    deposit_records_alg_1 = get_deposit_records(bounding_box, minerals)
    deposit_records_alg_2 = get_deposit_records(bounding_box, minerals, kd_tree)

    # Use naive method to find mineral deposits in bounding box.
    deposit_records_naive = []
    for i in range(0, len(minerals)):
        record = minerals[i]
        lat = record[1]
        lon = record[2]
        if (lat_south < lat and lat_north > lat and lon_west < lon and lon_east > lon):
            deposit_records_naive.append([i, record])

    # Verify results are equal.
    print("Verifying results. . .")
    deposit_records_alg_1.sort(key=getRecordIndex)
    deposit_records_alg_2.sort(key=getRecordIndex)
    deposit_records_naive.sort(key=getRecordIndex)
    assert(deposit_records_alg_1 == deposit_records_alg_2)
    assert(deposit_records_alg_2 == deposit_records_naive)
    print("OK.")

    # Print countries of deposits.
    if print_countries:
        for item in deposit_records_naive:
            country = item[1][4]
            region = item[1][5]
            location = str(country) + ", " + str(region)
            print(location)

def test_against_known_deposits(ul, ur, ll, lr, minerals, kd_tree, expected_records):
    # For a convex 4-sided polygon, verify result against known deposits.

    # Create bounding box.
    bounding_box = {}
    bounding_box["CORNER_UL_LAT_PRODUCT"] = ul[0]
    bounding_box["CORNER_UL_LON_PRODUCT"] = ul[1]
    bounding_box["CORNER_UR_LAT_PRODUCT"] = ur[0]
    bounding_box["CORNER_UR_LON_PRODUCT"] = ur[1]
    bounding_box["CORNER_LL_LAT_PRODUCT"] = ll[0]
    bounding_box["CORNER_LL_LON_PRODUCT"] = ll[1]
    bounding_box["CORNER_LR_LAT_PRODUCT"] = lr[0]
    bounding_box["CORNER_LR_LON_PRODUCT"] = lr[1]

    # Use our algorithm, with and without kd tree, to find mineral deposits in bounding box.
    deposit_records_alg_1 = get_deposit_records(bounding_box, minerals)
    deposit_records_alg_2 = get_deposit_records(bounding_box, minerals, kd_tree)

    # Verify result equals expected.
    print("Verifying results. . .")
    assert(deposit_records_alg_1 == deposit_records_alg_2)
    assert(deposit_records_alg_2 == expected_records)
    print("OK.")

def test_1():
    print("################################")
    print("TEST 1: test against naive method:")
    print_countries = False

    print("Loading minerals dataset. . .")
    minerals, minerals_cleaned, minerals_kd_tree, minerals_cleaned_kd_tree = load_all_minerals_and_kd_trees()
    print("Loaded " + str(len(minerals)) + " rows from minerals.csv.")
    print("Loaded " + str(len(minerals_cleaned)) + " rows from minerals_cleaned.csv.")

    print("Testing dummy bounding box: Wyoming. . .")
    lat_north = 44.6854
    lat_south = 41.2457
    lon_west = -110.7871
    lon_east = -104.4551
    test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals_cleaned, minerals_cleaned_kd_tree, print_countries)
    test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals, minerals_kd_tree, print_countries)

    print("Testing dummy bounding box: Western Australia. . .")
    lat_north = -22.2031
    lat_south = -31.1178
    lon_west = 115.2840
    lon_east = 127.4519
    test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals_cleaned, minerals_cleaned_kd_tree, print_countries)
    test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals, minerals_kd_tree, print_countries)

    print("Testing dummy bounding box: Congo. . .")
    lat_north = 36.5799
    lat_south = 32.5188
    lon_west = 8.5227
    lon_east = 11.2603
    test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals_cleaned, minerals_cleaned_kd_tree, print_countries)
    test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals, minerals_kd_tree, print_countries)

def test_2():
    print("################################")
    print("TEST 2: test against known deposits:")

    print("Loading minerals dataset. . .")
    minerals = load_minerals("minerals_cleaned.csv")
    minerals_kd_tree = get_kd_tree(minerals)
    print("Loaded " + str(len(minerals)) + " rows from minerals_cleaned.csv.")

    print("Testing location 1. . .")
    expected = [
        [1235, ["Bougrine", 36.49944, 8.50208, "AF", "Tunisia", None, None, "B", "Lead", "Zinc", "Fluorine-Fluorite, Barium-Barite", "Surface", None, "Producer", None, None, "Underground", None, None]]
    ]

    print("Square region, aligned with lat/lon axes. . .")
    ul = [36.6, 8.4]
    ur = [36.6, 8.6]
    ll = [36.4, 8.4]
    lr = [36.4, 8.6]
    test_against_known_deposits(ul, ur, ll, lr, minerals, minerals_kd_tree, expected)

    print("Square region, skewed. . .")
    ul = [36.6, 8.5]
    ur = [36.5, 8.6]
    ll = [36.5, 8.4]
    lr = [36.4, 8.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, minerals_kd_tree, expected)

    print("Square region, skewed further. . .")
    ul = [36.4, 8.5]
    ur = [36.5, 8.4]
    ll = [36.5, 8.6]
    lr = [36.6, 8.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, minerals_kd_tree, expected)

    print("Vertically elongated square region, skewed. . .")
    ul = [36.7, 8.5]
    ur = [36.5, 8.6]
    ll = [36.5, 8.4]
    lr = [36.3, 8.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, minerals_kd_tree, expected)

    print("Horizontally elongated square region, skewed. . .")
    ul = [36.6, 8.5]
    ur = [36.5, 8.7]
    ll = [36.5, 8.3]
    lr = [36.4, 8.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, minerals_kd_tree, expected)

    print("Testing location 2. . .")
    expected = [
        [1686, ["Caracota Mine", -20.08089, -65.91615, "SA", "Bolivia", "La Paz", None, "M", "Antimony", None, "Gold", "Underground", None, "Producer", None, None, "Underground", None, None]]
    ]
    print("Square region, aligned with lat/lon axes. . .")
    ul = [-20.0, -66.0]
    ur = [-20.0, -65.8]
    ll = [-20.2, -66.0]
    lr = [-20.2, -65.8]
    test_against_known_deposits(ul, ur, ll, lr, minerals, minerals_kd_tree, expected)

    print("Testing location 3. . .")
    expected = [
        [1708, ["Discovery Bay Water Valley", 18.34788, -77.39004, "CR", "Jamaica", None, None, "M", "Aluminum", None, "Silica", "Surface", None, "Producer", None, None, "Surface", None, None]]
    ]
    print("Square region, skewed. . .")
    ul = [18.45, -77.4]
    ur = [18.35, -77.3]
    ll = [18.35, -77.5]
    lr = [18.25, -77.4]
    test_against_known_deposits(ul, ur, ll, lr, minerals, minerals_kd_tree, expected)

    print("Testing location 4. . .")
    expected = [
        [1767, ["Lagonoy High Grade", 13.91694, 123.49924, "AS", "Philippines", None, None, "M", "Chromium", None, None, "Surface", None, "Producer", None, None, "Underground", None, None]]
    ]
    print("Square region, skewed further. . .")
    ul = [13.8, 123.5]
    ur = [13.9, 123.4]
    ll = [13.9, 123.6]
    lr = [14.0, 123.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, minerals_kd_tree, expected)

    print("Testing location 5. . .")
    expected = [
        [1814, ["Jin He Phosphate Mine", 29.16557, 103.333, "AS", "China", "Sichuan [Szechwan]", None, "N", "Phosphorus-Phosphates", None, None, "Underground", None, "Producer", None, None, "Underground", None, None]]
    ]
    print("Vertically elongated square region, skewed. . .")
    ul = [29.45, 103.3]
    ur = [29.15, 103.4]
    ll = [29.15, 103.2]
    lr = [28.85, 103.3]
    test_against_known_deposits(ul, ur, ll, lr, minerals, minerals_kd_tree, expected)

def test_3():
    # Compare search performance with and without kd tree.

    print("################################")
    print("TEST 3: kd tree performance test:")

    print("Loading minerals dataset. . .")
    minerals, minerals_cleaned, minerals_kd_tree, minerals_cleaned_kd_tree = load_all_minerals_and_kd_trees()
    print("Loaded " + str(len(minerals)) + " rows from minerals.csv.")
    print("Loaded " + str(len(minerals_cleaned)) + " rows from minerals_cleaned.csv.")

    # Create dummy bounding box: Wyoming.
    ul = [44.6854, -110.7871]
    ur = [44.6854, -104.4551]
    ll = [41.2457, -110.7871]
    lr = [41.2457, -104.4551]
    bounding_box = {}
    bounding_box["CORNER_UL_LAT_PRODUCT"] = ul[0]
    bounding_box["CORNER_UL_LON_PRODUCT"] = ul[1]
    bounding_box["CORNER_UR_LAT_PRODUCT"] = ur[0]
    bounding_box["CORNER_UR_LON_PRODUCT"] = ur[1]
    bounding_box["CORNER_LL_LAT_PRODUCT"] = ll[0]
    bounding_box["CORNER_LL_LON_PRODUCT"] = ll[1]
    bounding_box["CORNER_LR_LAT_PRODUCT"] = lr[0]
    bounding_box["CORNER_LR_LON_PRODUCT"] = lr[1]

    print("Testing on minerals_cleaned without kd tree. . .")
    num_queries = 1000
    start_without_kd = time.time()
    results_without_kd = []
    for i in range(0, num_queries):
        results_without_kd.append(get_deposit_records(bounding_box, minerals_cleaned))
    elapsed_without_kd = time.time() - start_without_kd

    print("Testing minerals_cleaned with kd tree. . .")
    start_with_kd = time.time()
    results_with_kd = []
    for i in range(0, num_queries):
        results_with_kd.append(get_deposit_records(bounding_box, minerals_cleaned, minerals_cleaned_kd_tree))
    elapsed_with_kd = time.time() - start_with_kd

    print("Without kd tree: generated " + str(len(results_without_kd)) + " labels in " + str(elapsed_without_kd) + " seconds.")
    print("With kd tree: generated " + str(len(results_with_kd)) + " labels in " + str(elapsed_with_kd) + " seconds.")

    print("Verifying results. . .")
    assert(len(results_with_kd) == len(results_without_kd))
    for i in range(len(results_without_kd)):
        results_without_kd[i].sort(key=getRecordIndex)
        results_with_kd[i].sort(key=getRecordIndex)
    assert(results_without_kd == results_with_kd)
    print("OK.")

    print("Testing on minerals without kd tree. . .")
    num_queries = 50
    start_without_kd = time.time()
    results_without_kd = []
    for i in range(0, num_queries):
        results_without_kd.append(get_deposit_records(bounding_box, minerals))
    elapsed_without_kd = time.time() - start_without_kd

    print("Testing minerals with kd tree. . .")
    start_with_kd = time.time()
    results_with_kd = []
    for i in range(0, num_queries):
        results_with_kd.append(get_deposit_records(bounding_box, minerals, minerals_kd_tree))
    elapsed_with_kd = time.time() - start_with_kd

    print("Without kd tree: generated " + str(len(results_without_kd)) + " labels in " + str(elapsed_without_kd) + " seconds.")
    print("With kd tree: generated " + str(len(results_with_kd)) + " labels in " + str(elapsed_with_kd) + " seconds.")

    print("Verifying results. . .")
    assert(len(results_with_kd) == len(results_without_kd))
    for i in range(len(results_without_kd)):
        results_without_kd[i].sort(key=getRecordIndex)
        results_with_kd[i].sort(key=getRecordIndex)
    assert(results_without_kd == results_with_kd)
    print("OK.")

    return

def test_4():
    # Performance test: end-to-end:

    print("################################")
    print("TEST 4: end-to-end performance test:")
    num_labels = 1000

    print("Loading minerals dataset. . .")
    minerals, minerals_cleaned, minerals_kd_tree, minerals_cleaned_kd_tree = load_all_minerals_and_kd_trees()
    print("Loaded " + str(len(minerals)) + " rows from minerals.csv.")
    print("Loaded " + str(len(minerals_cleaned)) + " rows from minerals_cleaned.csv.")

    print("Generating labels using minerals_cleaned.csv. . .")
    metadata_filename = "LT04_L1GS_001026_19821202_20170220_01_T2_MTL.txt"
    metadata_filepath = os.path.join("data", "metadata", metadata_filename)
    start = time.time()
    labels_minerals_cleaned = []
    for i in range(0, num_labels):
        labels_minerals_cleaned.append(generate_label(metadata_filepath, minerals_cleaned, minerals_cleaned_kd_tree, label_type="binary"))
    elapsed_minerals_cleaned = time.time() - start

    print("Generating labels using minerals.csv. . .")
    start = time.time()
    labels_minerals = []
    for i in range(0, num_labels):
        labels_minerals.append(generate_label(metadata_filepath, minerals, minerals_kd_tree, label_type="binary"))
    elapsed_minerals = time.time() - start

    print("Using minerals_cleaned.csv: generated " + str(len(labels_minerals_cleaned)) + " labels in " + str(elapsed_minerals_cleaned) + " seconds.")
    print("Using minerals.csv: generated " + str(len(labels_minerals)) + " labels in " + str(elapsed_minerals) + " seconds.")


################################################################
# Run program.
################################################################

generate_and_store_all_labels()


################################################################
# Run tests.
################################################################

# test_1()
# test_2()
# test_3()
# test_4()