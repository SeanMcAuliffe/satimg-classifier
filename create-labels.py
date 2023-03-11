# Create labels for landsat images.

import pandas as pd
import numpy as np
import os.path

################################################################
# Configuration.
################################################################

# label_type = "binary"
label_type = "multi-class"
# minerals_filename = "minerals.csv"
minerals_filename = "minerals_cleaned.csv"

################################################################
# Functions.
################################################################

def load_minerals():
    # Assumes csv files are in directory /data/minerals/
    # "minerals" is a list of lists, where each sublist is a row of the csv.
    minerals_path = os.path.join("data", "minerals", minerals_filename)
    minerals_rows = pd.read_csv(minerals_path, header=0, low_memory=False)
    minerals_rows = minerals_rows.replace({np.nan:None})
    return minerals_rows.values.tolist()

def get_bounding_box(md_filename):
    # From metadata file, obtain coords of bounding box of image.
    bounding_box = {}
    with open(md_filename) as md_file:
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

def get_deposit_records(bounding_box, minerals):
    # Using image bounding box, find all mineral deposits present in the image.
    # Returns list of lists each of form [index, record].

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

    # Find deposits within bounding box.
    deposits = [] # Has form [index, record]
    for j in range(0, len(minerals)):
        deposit = minerals[j]
        deposit_coords = [deposit[1], deposit[2]]
        deposit_in_img = True
        # Image is a convex polygon. Then if deposit is on right-hand side of each edge, deposit is in the image.
        # TODO: implement spatial data structure to accelerate.
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
            deposits.append([j, minerals[j]])
    return deposits

def convert_to_multiclass_label(deposit_records):
    # TODO: convert a list of deposit records into a multiclass label indicating which minerals are present.
    return

def generate_label(md_filename, minerals):
    # From metadata filename and minerals dataset, create label for image.
    bounding_box = get_bounding_box(md_filename)
    deposit_records = get_deposit_records(bounding_box, minerals)

    if label_type == "binary":
        label = [0] if len(deposit_records) == 0 else [1]
    else:
        label = deposit_records
        # label = convert_to_multiclass_label(deposit_records)
        
    return label


################################################################
# Tests.
################################################################

def test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals, print_countries=False):
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

    # Use our algorithm to find mineral deposits in bounding box.
    deposit_records_alg = get_deposit_records(bounding_box, minerals)

    # Use naive method to find mineral deposits in bounding box.
    deposit_records_alt = []
    for i in range(0, len(minerals)):
        record = minerals[i]
        lat = record[1]
        lon = record[2]
        if (lat_south < lat and lat_north > lat and lon_west < lon and lon_east > lon):
            deposit_records_alt.append([i, record])

    # Verify results are equal.
    print("Verifying results. . .")
    def getRecordIndex(r):
        return r[0]
    deposit_records_alg.sort(key=getRecordIndex)
    deposit_records_alt.sort(key=getRecordIndex)
    assert(deposit_records_alg == deposit_records_alt)
    print("OK.")

    # Print countries of deposits.
    if print_countries:
        for item in deposit_records_alg:
            country = item[1][4]
            region = item[1][5]
            location = str(country) + ", " + str(region)
            print(location)

def test_against_known_deposits(ul, ur, ll, lr, minerals, expected_records):
    # For a convex 4-sided polygon, verify result against known deposits.

    # Create dummy, square bounding box:
    bounding_box = {}
    bounding_box["CORNER_UL_LAT_PRODUCT"] = ul[0]
    bounding_box["CORNER_UL_LON_PRODUCT"] = ul[1]
    bounding_box["CORNER_UR_LAT_PRODUCT"] = ur[0]
    bounding_box["CORNER_UR_LON_PRODUCT"] = ur[1]
    bounding_box["CORNER_LL_LAT_PRODUCT"] = ll[0]
    bounding_box["CORNER_LL_LON_PRODUCT"] = ll[1]
    bounding_box["CORNER_LR_LAT_PRODUCT"] = lr[0]
    bounding_box["CORNER_LR_LON_PRODUCT"] = lr[1]

    # Use our algorithm to find mineral deposits in bounding box.
    deposit_records_alg = get_deposit_records(bounding_box, minerals)

    # Verify result equals expected.
    print("Verifying results. . .")
    assert(deposit_records_alg == expected_records)
    print("OK.")

def test_1():
    # Expects "minerals_cleaned.csv".
    print("################################")
    print("TEST 1: test against naive method:")
    print_countries = False

    print("Loading minerals dataset. . .")
    minerals = load_minerals()
    print("Loaded " + str(len(minerals)) + " rows from " + minerals_filename + ".")

    print("Testing dummy bounding box: Wyoming. . .")
    lat_north = 44.6854
    lat_south = 41.2457
    lon_west = -110.7871
    lon_east = -104.4551
    test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals, print_countries)

    print("Testing dummy bounding box: Western Australia. . .")
    lat_north = -22.2031
    lat_south = -31.1178
    lon_west = 115.2840
    lon_east = 127.4519
    test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals, print_countries)

    print("Testing dummy bounding box: Congo. . .")
    lat_north = 36.5799
    lat_south = 32.5188
    lon_west = 8.5227
    lon_east = 11.2603
    test_against_naive_method(lat_north, lat_south, lon_west, lon_east, minerals, print_countries)

def test_2():
    # Expects "minerals_cleaned.csv".
    print("################################")
    print("TEST 2: test against known deposits:")

    print("Loading minerals dataset. . .")
    minerals = load_minerals()
    print("Loaded " + str(len(minerals)) + " rows from " + minerals_filename + ".")

    print("Testing location 1. . .")
    expected = [
        [1235, ["Bougrine", 36.49944, 8.50208, "AF", "Tunisia", None, None, "B", "Lead", "Zinc", "Fluorine-Fluorite, Barium-Barite", "Surface", None, "Producer", None, None, "Underground", None, None]]
    ]

    print("Square region, aligned with lat/lon axes. . .")
    ul = [36.6, 8.4]
    ur = [36.6, 8.6]
    ll = [36.4, 8.4]
    lr = [36.4, 8.6]
    test_against_known_deposits(ul, ur, ll, lr, minerals, expected)

    print("Square region, skewed. . .")
    ul = [36.6, 8.5]
    ur = [36.5, 8.6]
    ll = [36.5, 8.4]
    lr = [36.4, 8.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, expected)

    print("Square region, skewed further. . .")
    ul = [36.4, 8.5]
    ur = [36.5, 8.4]
    ll = [36.5, 8.6]
    lr = [36.6, 8.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, expected)

    print("Vertically elongated square region, skewed. . .")
    ul = [36.7, 8.5]
    ur = [36.5, 8.6]
    ll = [36.5, 8.4]
    lr = [36.3, 8.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, expected)

    print("Horizontally elongated square region, skewed. . .")
    ul = [36.6, 8.5]
    ur = [36.5, 8.7]
    ll = [36.5, 8.3]
    lr = [36.4, 8.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, expected)

    print("Testing location 2. . .")
    expected = [
        [1686, ["Caracota Mine", -20.08089, -65.91615, "SA", "Bolivia", "La Paz", None, "M", "Antimony", None, "Gold", "Underground", None, "Producer", None, None, "Underground", None, None]]
    ]
    print("Square region, aligned with lat/lon axes. . .")
    ul = [-20.0, -66.0]
    ur = [-20.0, -65.8]
    ll = [-20.2, -66.0]
    lr = [-20.2, -65.8]
    test_against_known_deposits(ul, ur, ll, lr, minerals, expected)

    print("Testing location 3. . .")
    expected = [
        [1708, ["Discovery Bay Water Valley", 18.34788, -77.39004, "CR", "Jamaica", None, None, "M", "Aluminum", None, "Silica", "Surface", None, "Producer", None, None, "Surface", None, None]]
    ]
    print("Square region, skewed. . .")
    ul = [18.45, -77.4]
    ur = [18.35, -77.3]
    ll = [18.35, -77.5]
    lr = [18.25, -77.4]
    test_against_known_deposits(ul, ur, ll, lr, minerals, expected)

    print("Testing location 4. . .")
    expected = [
        [1767, ["Lagonoy High Grade", 13.91694, 123.49924, "AS", "Philippines", None, None, "M", "Chromium", None, None, "Surface", None, "Producer", None, None, "Underground", None, None]]
    ]
    print("Square region, skewed further. . .")
    ul = [13.8, 123.5]
    ur = [13.9, 123.4]
    ll = [13.9, 123.6]
    lr = [14.0, 123.5]
    test_against_known_deposits(ul, ur, ll, lr, minerals, expected)

    print("Testing location 5. . .")
    expected = [
        [1814, ["Jin He Phosphate Mine", 29.16557, 103.333, "AS", "China", "Sichuan [Szechwan]", None, "N", "Phosphorus-Phosphates", None, None, "Underground", None, "Producer", None, None, "Underground", None, None]]
    ]
    print("Vertically elongated square region, skewed. . .")
    ul = [29.45, 103.3]
    ur = [29.15, 103.4]
    ll = [29.15, 103.2]
    lr = [28.85, 103.3]
    test_against_known_deposits(ul, ur, ll, lr, minerals, expected)


################################################################
# Run program.
################################################################

test_1()
test_2()