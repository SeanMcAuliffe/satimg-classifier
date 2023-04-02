from math import floor 
from PIL import Image
import numpy as np
import random
import os
from copy import deepcopy
import tifffile
import gc
from math import log
from scipy.stats import skew
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from utils import get_bounding_box, analyze_bounding_box, plot_on_world_map
    from preprocessing import downsample_to
    from utils import ImageBucket
else:
    from utils.utils import get_bounding_box, analyze_bounding_box, plot_on_world_map_pos_neg
    from utils.preprocessing import downsample_to
    from utils.utils import ImageBucket


def load_datasets(total_images: int = 10000, train_proportion: float = 0.8,
                  remove_ocean: bool = True, normalize: bool = True,
                  downscale_dimension: int = 512):
    
    """ Load the datasets from the images and labels files. """

    IMG_DIRPATH = os.path.join("..", "data", "images_norm")
    META_DIRPATH = os.path.join("..", "data", "metadata")
    LABELS_FILEPATH= os.path.join("..", "data", "labels", "labels_binary_minerals.csv")
    BRIGHTNESS_FILEPATH = os.path.join("..", "data", "labels", "brightness.csv")
    all_images = {}
    all_labels = {}
    all_coords = {}
    all_brightnesses = {}

    print("Loading image data ...")
    for image_name in os.listdir(IMG_DIRPATH):
        all_images[image_name[:-7]] = None

    print("Loading label data ...")
    with open(LABELS_FILEPATH, 'r') as f:
        f.readline()
        for line in f.readlines():
            name, label = line.rstrip().split(',')
            all_labels[name[:-4]] = int(label)

    print("Loading coords ...")
    for meta_name in os.listdir(META_DIRPATH):
        m_path = os.path.join(META_DIRPATH, meta_name)
        bb = get_bounding_box(m_path)
        ul, ur, ll, lr, edges = analyze_bounding_box(bb)
        midpoint = [(ul[0] + lr[0]) / 2.0, (ul[1] + lr[1]) / 2.0]
        imagename = f"{meta_name[:-8]}"
        all_coords[imagename] = midpoint

    print("Loading brightness values ...")
    with open(BRIGHTNESS_FILEPATH, 'r') as f:
        for line in f.readlines():
            name, label = line.rstrip().split(',')
            all_brightnesses[name] = float(label)
    
    # Split into buckets based on location
    location_buckets = {}
    keys = all_images.keys()
    for key in keys:
        tokenized = key.split('_')
        location_code = tokenized[2]
        if location_code not in location_buckets.keys():
            location_buckets[location_code] = []
        location_buckets[location_code].append(key)
    
    # Ensure all buckets are mono-label
    for location_code in location_buckets.keys():
        examples = location_buckets[location_code]
        examples = [[name, all_labels[name]] for name in examples]
        zeros = len([ex for ex in examples if ex[1] == 0])
        ones = len([ex for ex in examples if ex[1] == 1])
        if zeros != 0 and ones != 0:
            # We have multi-label
            if zeros < ones:
                # Remove those examples.
                examples = [ex for ex in examples if ex[1] == 1]
            else:
                examples = [ex for ex in examples if ex[1] == 0]
            location_buckets[location_code] = [x[0] for x in examples]

    keys_to_delete = []
    count = 0
    images_removed = 0
    if remove_ocean:
        print("Removing images over ocean from dataset ...")
        OCEANS_MASK_PATH = os.path.join("..", "data", "labels", "ocean_mask.csv")
        dry_examples = {}
        with open(OCEANS_MASK_PATH, 'r') as f:
            for line in f.readlines():
                name, label = line.rstrip().split(',')
                if int(label) == 1:
                    dry_examples[name] = True
                else:
                    dry_examples[name] = False
        for key in location_buckets.keys():
            over_land = dry_examples[location_buckets[key][0]]
            if not over_land:
                keys_to_delete.append(key)
                count += 1
                images_removed += len(location_buckets[key])

        for key in keys_to_delete:
            del location_buckets[key]

    print(f"Total Unique Buckets: {len(location_buckets.keys())}")

    # Separate positive and negative buckets.
    pos_buckets = {}
    neg_buckets = {}

    for code in location_buckets.keys():
        examples = location_buckets[code]
        label = all_labels[examples[0]]
        if label == 1:
            pos_buckets[code] = ImageBucket(code, examples, all_brightnesses)
        else:
            neg_buckets[code] = ImageBucket(code, examples, all_brightnesses)

    del location_buckets

    # Define bucket ranges for train, test.
    num_pos = len(pos_buckets.keys())
    num_neg = len(neg_buckets.keys())

    print(f"Total Neg. Buckets: {num_neg}")
    print(f"Total Pos. Buckets: {num_pos}")

    upper_pos_index = floor(num_pos * train_proportion)
    upper_neg_index = floor(num_neg * train_proportion)

    # Obtain train, test set.
    pos_keys = deepcopy(list(pos_buckets.keys()))
    neg_keys = deepcopy(list(neg_buckets.keys()))
    random.shuffle(pos_keys)
    random.shuffle(neg_keys)

    tr_names = []
    X_train = []
    Y_train = []

    val_names = []
    x_test = []
    y_test = []

    # pos_train_coords = []
    # neg_train_coords = []
    # pos_test_coords = []
    # neg_test_coords = []

    i = 0
    print("Sampling training set ...")
    while len(X_train) < (total_images * train_proportion):
        p_bucket_key = pos_keys[i % upper_pos_index]
        n_bucket_key = neg_keys[i % upper_neg_index]
        p_bucket = pos_buckets[p_bucket_key]
        n_bucket = neg_buckets[n_bucket_key]
        p_example_name = p_bucket.get_next_image()
        n_example_name = n_bucket.get_next_image()
        p_path = os.path.join(IMG_DIRPATH, f'{p_example_name}_B7.TIF')
        n_path = os.path.join(IMG_DIRPATH, f'{n_example_name}_B7.TIF')
        p_im = np.array(Image.open(p_path))
        n_im = np.array(Image.open(n_path))
        if downscale_dimension < 512:
            p_im = downsample_to(p_im, downscale_dimension)
            n_im = downsample_to(n_im, downscale_dimension)
        if normalize:
            p_im = np.divide(p_im, 255.0)
            n_im = np.divide(n_im, 255.0)
        X_train.append(p_im)
        X_train.append(n_im)
        Y_train.append(all_labels[p_example_name])
        Y_train.append(all_labels[n_example_name])
        tr_names.append(p_example_name)
        tr_names.append(n_example_name)
        i += 1

        # pos_train_coords.append(all_coords[p_example_name])
        # neg_train_coords.append(all_coords[n_example_name])

    i = 0
    print("Sampling test set ...")
    p_num = len(pos_keys[upper_pos_index:])
    n_num = len(neg_keys[upper_neg_index:])
    while len(x_test) < (total_images * (1 - train_proportion)):
        p_bucket_key = pos_keys[upper_pos_index + (i % p_num)]
        n_bucket_key = neg_keys[upper_neg_index + (i % n_num)] 
        p_bucket = pos_buckets[p_bucket_key]
        n_bucket = neg_buckets[n_bucket_key]
        p_example_name = p_bucket.get_next_image()
        n_example_name = n_bucket.get_next_image()
        p_path = os.path.join(IMG_DIRPATH, f'{p_example_name}_B7.TIF')
        n_path = os.path.join(IMG_DIRPATH, f'{n_example_name}_B7.TIF')
        p_im = np.array(Image.open(p_path))
        n_im = np.array(Image.open(n_path))
        if downscale_dimension < 512:
            p_im = downsample_to(p_im, downscale_dimension)
            n_im = downsample_to(n_im, downscale_dimension)
        if normalize:
            p_im = np.divide(p_im, 255.0)
            n_im = np.divide(n_im, 255.0)
        x_test.append(p_im)
        x_test.append(n_im)
        y_test.append(all_labels[p_example_name])
        y_test.append(all_labels[n_example_name])
        val_names.append(p_example_name)
        val_names.append(n_example_name)
        i += 1

        # pos_test_coords.append(all_coords[p_example_name])
        # neg_test_coords.append(all_coords[n_example_name])

    del neg_keys
    del pos_keys
    del upper_pos_index
    del upper_neg_index
    del num_pos
    del num_neg
    del keys_to_delete
    del pos_buckets
    del neg_buckets
    del all_images
    del all_labels
    del all_coords
    del all_brightnesses
    del keys
    gc.collect()

    # if __name__ == "__main__":
    #     return pos_train_coords, neg_train_coords, pos_test_coords, neg_test_coords
    # else:
    print("Converting to nd.arrays")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print("Datasets created successfully.")
    return X_train, Y_train, x_test, y_test, tr_names, val_names

# ------------------------------------------------------------------------------

def load_datasets_reg(total_images: int = 10000, train_proportion: float = 0.8,
                  remove_ocean: bool = True, normalize: bool = True,
                  downscale_dimension: int = 512):
    
    """ Load the datasets from the images and labels files. """

    IMG_DIRPATH = os.path.join("..", "data", "images_norm")
    META_DIRPATH = os.path.join("..", "data", "metadata")
    LABELS_FILEPATH= os.path.join("..", "data", "labels", "labels_binary_minerals_reg.csv")
    BRIGHTNESS_FILEPATH = os.path.join("..", "data", "labels", "brightness.csv")
                                       
    all_images = {}
    all_labels = {}
    all_coords = {}
    all_brightnesses = {}

    print("Loading image data ...")
    for image_name in os.listdir(IMG_DIRPATH):
        all_images[image_name[:-7]] = None

    print("Loading label data ...")
    with open(LABELS_FILEPATH, 'r') as f:
        f.readline()
        for line in f.readlines():
            name, label = line.rstrip().split(',')
            all_labels[name[:-4]] = float(label)

    print("Loading coords ...")
    for meta_name in os.listdir(META_DIRPATH):
        m_path = os.path.join(META_DIRPATH, meta_name)
        bb = get_bounding_box(m_path)
        ul, ur, ll, lr, edges = analyze_bounding_box(bb)
        midpoint = [(ul[0] + lr[0]) / 2.0, (ul[1] + lr[1]) / 2.0]
        imagename = f"{meta_name[:-8]}"
        all_coords[imagename] = midpoint

    print("Loading brightness values ...")
    with open(BRIGHTNESS_FILEPATH, 'r') as f:
        for line in f.readlines():
            name, label = line.rstrip().split(',')
            all_brightnesses[name] = float(label)
    
    # Split into buckets based on location
    location_buckets = {}
    keys = all_images.keys()
    for key in keys:
        tokenized = key.split('_')
        location_code = tokenized[2]
        if location_code not in location_buckets.keys():
            location_buckets[location_code] = []
        location_buckets[location_code].append(key)
    
    # Ensure all buckets are resourcey
    keys_to_delete = []
    for code in location_buckets.keys():
        name_list = location_buckets[code]
        richness_scores_lst = [all_labels[name] for name in name_list]
        total_bck_richness = sum(richness_scores_lst)
        if total_bck_richness == 0:
            keys_to_delete.append(code)

    # Remove all buckets with no richness
    for key in keys_to_delete:
        del location_buckets[key]
      
    # Remove buckets over ocean.
    keys_to_delete = []
    if remove_ocean:
        print("Removing images over ocean from dataset ...")
        OCEANS_MASK_PATH = os.path.join("..", "data", "labels", "ocean_mask.csv")
        dry_examples = {}
        with open(OCEANS_MASK_PATH, 'r') as f:
            for line in f.readlines():
                name, label = line.rstrip().split(',')
                if int(label) == 1:
                    dry_examples[name] = True
                else:
                    dry_examples[name] = False
        for key in location_buckets.keys():
            over_land = dry_examples[location_buckets[key][0]]
            if not over_land:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del location_buckets[key]

    print(f"Total Rich+Dry Buckets: {len(location_buckets.keys())}")

    # Split buckets into lists of training and test buckets
    
    # Randomly sample from the buckets until we have the desired number of images.

    buckets = {}

    for code in location_buckets.keys():
        examples = location_buckets[code]
        buckets[code] = ImageBucket(code, examples, all_brightnesses)

    keys = list(buckets.keys())
    random.shuffle(keys)

    # Split the keys into two lists

    tr_index = floor(len(keys) * train_proportion)
    
    tr_buckets = keys[:tr_index]
    te_buckets = keys[tr_index:]

    random.shuffle(tr_buckets)
    random.shuffle(te_buckets)

    tr_names = []
    X_train = []
    Y_train = []

    val_names = []
    x_test = []
    y_test = []

    i = 0
    print("Sampling training set ...")
    while len(X_train) < (total_images * train_proportion):
        bucket_key = tr_buckets[i % len(tr_buckets)]
        bucket = buckets[bucket_key]
        example_name = bucket.get_next_image()
        path = os.path.join(IMG_DIRPATH, f'{example_name}_B7.TIF')
        im = np.array(Image.open(path))
        if downscale_dimension < 512:
            im = downsample_to(im, downscale_dimension)
        if normalize:
            im = np.divide(im, 255.0) # TODO: Check if this is the right way to normalize
        X_train.append(im)
        Y_train.append(all_labels[example_name])
        tr_names.append(example_name)
        i += 1

    # imbalance = skew(Y_train)
    # print("Train Skew before log-transformation: ", imbalance)
    # # Apply log transformation.
    # Y_train = [log(label + 1.0) for label in Y_train]
    # greatest = max(Y_train)
    # Y_train = [label / greatest for label in Y_train]

    # imbalance = skew(Y_train)
    # print("Train Skew after log-transformation: ", imbalance)

    # plt.figure(figsize=(8, 4))
    # plt.hist(Y_train, bins=10, edgecolor='k', alpha=0.7)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram')
    # plt.show()
    #random.shuffle(Y_train)

    i = 0
    print("Sampling test set ...")
    while len(x_test) < (total_images * (1 - train_proportion)):
        bucket_key = te_buckets[i % len(te_buckets)]
        bucket = buckets[bucket_key]
        example_name = bucket.get_next_image()
        path = os.path.join(IMG_DIRPATH, f'{example_name}_B7.TIF')
        im = np.array(Image.open(path))
        if downscale_dimension < 512:
            im = downsample_to(im, downscale_dimension)
        if normalize:
            im = np.divide(im, 255.0)
        x_test.append(im)
        y_test.append(all_labels[example_name])
        val_names.append(example_name)
        i += 1


    plt.figure(figsize=(8, 4))
    plt.hist(y_test, bins=10, edgecolor='k', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()
    # imbalance = skew(y_test)
    # print("Test Skew before log-transformation: ", imbalance)
    # # Apply log transformation.
    # y_test = [log(label + 1.0) for label in y_test]
    # greatest = max(y_test)
    # y_test = [label / greatest for label in y_test]

    # imbalance = skew(y_test)
    # print("Test Skew after log-transformation: ", imbalance)

    return (np.array(X_train), np.array(Y_train), np.array(x_test), np.array(y_test), tr_names, val_names)
    
   
if __name__ == "__main__":
    # Below is for testing purposes
    p_tr, n_tr, p_te, n_te = load_datasets()

    p_tr = [[point[1], point[0]] for point in p_tr]
    n_tr = [[point[1], point[0]] for point in n_tr]
    p_te = [[point[1], point[0]] for point in p_te]
    n_te = [[point[1], point[0]] for point in n_te]

    title1 = "train: pos, neg"
    filename1 = "world-map-train.png"
    plot_on_world_map_pos_neg(p_tr, n_tr, title=title1, xlabel="longitude", ylabel="latitude", filename=filename1)

    title2 = "test: pos, neg"
    filename2 = "world-map-test.png"
    plot_on_world_map_pos_neg(p_te, n_te, title=title2, xlabel="longitude", ylabel="latitude", filename=filename2)

