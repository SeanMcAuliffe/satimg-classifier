import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import skimage.io as skio
from skimage.transform import resize 
import os
import pandas as pd
from PIL import Image
from keras.initializers import HeUniform, HeNormal
import random
from keras import regularizers
from keras import backend as K
import gc
import sys
import matplotlib.image as mpimg
import math
import copy

################################################################
# Initialize paths.
print("Initializing paths...")
metadata_dir_path = os.path.join("data", "metadata")
labels_dir_path = os.path.join("data", "labels")
vis_dir_path = os.path.join("vis")
image_dir_path = os.path.join("data", "images", "B7_512x512", "archive")

################################################################
# Get batchname from command line, if provided.
print("Getting labels CSV filename from command line...")
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

################################################################
# Load labels.
# - Create dict of form (imagename, label)
print("Loading labels...")
labels_bm = []

for batch_name in os.listdir(labels_dir_path):
    if user_batch_name is not None and batch_name != user_batch_name:
        continue

    batch_path = os.path.join(metadata_dir_path, batch_name)
    labels_batch_path = os.path.join(labels_dir_path, batch_name)

    labels_bm_file_path = os.path.join(labels_batch_path, "labels_binary_minerals.csv")
    
    labels_bm_rows = pd.read_csv(labels_bm_file_path, header=0, low_memory=False)

    labels_bm += labels_bm_rows.values.tolist()

labels_bm = dict(labels_bm)

################################################################
# Load metadata.
# - Create dict of form (imagename, midpoint coords)
print("Loading metadata...")
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

# For each example. . .
coords = {}
for batch_name in os.listdir(metadata_dir_path):
    if user_batch_name is not None and batch_name != user_batch_name:
        continue

    batch_path = os.path.join(metadata_dir_path, batch_name)
    for filename in os.listdir(batch_path):
        metadata_file_path = os.path.join(batch_path, filename)

        # Obtain midpoint of image.
        bb = get_bounding_box(metadata_file_path)
        ul, ur, ll, lr, edges = analyze_bounding_box(bb)
        midpoint = [(ul[0] + lr[0]) / 2.0, (ul[1] + lr[1]) / 2.0]
        imagename = f"{filename[:-7]}MTL"
        coords[imagename] = midpoint

################################################################
# Load images.
# - Create dict of form (imagename, image)
print("Loading images...")

images = []
num_pixels = 262144 # 512 x 512
for i, imagename in enumerate(os.listdir(image_dir_path)):
   im = np.array(Image.open(os.path.join(image_dir_path, imagename)))
   im = im.flatten()
   im_pixels = len(im)
   difference = num_pixels - im_pixels

   if difference > 0:
      zeros = np.zeros((difference, 1), dtype=np.uint8)
      im = np.append(im, zeros)

   im = im.reshape((512, 512))

   if im is not None:
      imagename = f"{imagename[:-6]}MTL"
      images.append([imagename, im])

################################################################
# Create X_train, y_train, X_test, y_test.
# Ensure no location overlap exists between training and testing sets.
print("Creating training and testing sets. . .")

# Collect all loaded image data.
# - Create tuples of form (imagename, image, coords, label)
images_ext = []
for im in images:
    im += [coords[im[0]]]
    im += [labels_bm[im[0]]]
    images_ext.append(im)

# Group images taken approximately at the same location.
image_sets = []
while len(images_ext) > 0:
    i1 = images_ext.pop()
    i1_coords = i1[2]
    image_set = []
    image_set.append(i1)
    for i, i2 in enumerate(images_ext):
        i2_coords = i2[2]
        if math.dist(i1_coords, i2_coords) < 1.0:
            image_set.append(images_ext.pop(i))
    image_sets.append(image_set)

# Remove groups containing images having different labels.
# This occurs if some images of the group are positioned such that they contain a particular deposit.
# This removes only a small fraction of sets.
for i, iset in enumerate(image_sets):
    label1 = iset[0][3]
    for im in iset:
        label2 = im[3]
        if label2 != label1:
            image_sets.pop(i)

# Organize into positive and negative image sets.
positive_image_sets = [iset for iset in image_sets if iset[0][3] == 1]
negative_image_sets = [iset for iset in image_sets if iset[0][3] == 0]

# Separate into train and test sets having approximately the same proportion of positive and negative images.
# i.e. ensure both sets have approximately same class distribution.
# Also store coords of pos and neg examples for later use in visualizations.
X_train = []
y_train = []
X_test = []
y_test = []
train_pos_coords = []
train_neg_coords = []
test_pos_coords = []
test_neg_coords = []
random.shuffle(positive_image_sets)
random.shuffle(negative_image_sets)
one_per_set = True
for i, iset in enumerate(positive_image_sets):
    if i < int(len(positive_image_sets) * 0.8):
        for image in iset:
            X_train.append(copy.deepcopy(image[1]))
            y_train.append(copy.deepcopy(image[3]))
            train_pos_coords.append(copy.deepcopy(image[2]))
            if one_per_set:
                break
    else:
        for image in iset:
            X_test.append(copy.deepcopy(image[1]))
            y_test.append(copy.deepcopy(image[3]))
            test_pos_coords.append(copy.deepcopy(image[2]))
            if one_per_set:
                break
            
for i, iset in enumerate(negative_image_sets):
    if i < int(len(negative_image_sets) * 0.8):
        for image in iset:
            X_train.append(copy.deepcopy(image[1]))
            y_train.append(copy.deepcopy(image[3]))
            train_neg_coords.append(copy.deepcopy(image[2]))
            if one_per_set:
                break
    else:
        for image in iset:
            X_test.append(copy.deepcopy(image[1]))
            y_test.append(copy.deepcopy(image[3]))
            test_neg_coords.append(copy.deepcopy(image[2]))
            if one_per_set:
                break

# Convert images to inputs of format expected by Tensorflow.
# https://www.tensorflow.org/tutorials/images/cnn

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Summarize data.
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
print("Num pos train examples: ", len(train_pos_coords))
print("Num neg train examples: ", len(train_neg_coords))
print("Num pos test examples: ", len(test_pos_coords))
print("Num neg test examples: ", len(test_neg_coords))

# Free as much memory as possible.
del image_sets
del images_ext
del images
del coords
del labels_bm
del positive_image_sets
del negative_image_sets
gc.collect()

##############################################################
# MODEL:

# Create the model
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

print("Creating model. . .")

def train_and_evaluate_cnn(learning_rate):
   model = keras.Sequential()

   initializer = HeUniform(seed=random.randint(1, 100))

   model.add(keras.layers.Conv2D(filters=8,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_initializer=initializer,
                                 strides=(2, 2),
                                 input_shape=(512, 512, 1)))

   model.add(keras.layers.Conv2D(filters=16,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 strides=(2, 2),
                                 kernel_initializer=initializer))

   model.add(keras.layers.Conv2D(filters=32,
                                 kernel_size=(3, 3),
                                 strides=(2, 2),
                                 padding='same',
                                 activation='relu',
                                 kernel_initializer=initializer))

   # Flatten and put a fully connected layer.
   model.add(keras.layers.Flatten())
   model.add(keras.layers.BatchNormalization())

   # Dense layers.
   model.add(keras.layers.Dense(128, activation='relu'))
   model.add(keras.layers.Dense(64, activation='relu'))
   model.add(keras.layers.Dense(32, activation='relu'))

   # 1 output neuron for binary classification.
   model.add(keras.layers.Dense(1, activation='sigmoid'))
   model.summary()

   # Compile the model
   print("Compiling model. . .")
   opt = keras.optimizers.Adam(learning_rate=learning_rate)
   model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

   # Run CNN.
   print("Fitting model. . .")
   history = model.fit(X_train, y_train, verbose=1, epochs=100, batch_size=32, validation_data=(X_test, y_test))

   ################################################################
   # Evaluation:

   # Score the model
   print("Evaluating model. . .")
   test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1, batch_size=32)
   print("Test accuracy: ", test_acc)
   print("Test loss: ", test_loss)
   print(model.summary())

   del model

   return history

lrs = [0.000001, 0.00001, 0.0001]
histories = []
for lr in lrs:
   histories.append(train_and_evaluate_cnn(lr))
   gc.collect()
   K.clear_session()

##############################################################
# Visualize accuracy:
print("Visualizing accuracy. . .")
for i in range(0, len(histories)):
   p1 = plt.plot(histories[i].history['accuracy'], label = str(lrs[i]) + " train", linestyle='--')
   plt.plot(histories[i].history['val_accuracy'], label = str(lrs[i]) + " test", linestyle='-', color=p1[0].get_color())
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

#Save the plot to disc
plt.savefig(os.path.join("vis", "accuracy.png"), dpi=1200)

##############################################################
# Visualize train, test sets:
print("Visualizing train, test sets. . .")

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
train_pos_coords = [[point[1], point[0]] for point in train_pos_coords]
train_neg_coords = [[point[1], point[0]] for point in train_neg_coords]
test_pos_coords = [[point[1], point[0]] for point in test_pos_coords]
test_neg_coords = [[point[1], point[0]] for point in test_neg_coords]

# Plot on cartesian axes.
title1 = "train: pos, neg"
title1 += ", " + user_batch_name if user_batch_name is not None else ""
filename1 = "world-map-train"
filename1 += "-" + user_batch_name if user_batch_name is not None else ""
plot_on_world_map(train_pos_coords, train_neg_coords, title=title1, xlabel="longitude", ylabel="latitude")
plt.savefig(os.path.join(vis_dir_path, filename1), dpi=1200)
title2 = "test: pos, neg"
title2 += ", " + user_batch_name if user_batch_name is not None else ""
filename2 = "world-map-test"
filename2 += "-" + user_batch_name if user_batch_name is not None else ""
plot_on_world_map(test_pos_coords, test_neg_coords, title=title2, xlabel="longitude", ylabel="latitude")
plt.savefig(os.path.join(vis_dir_path, filename2), dpi=1200)

print("Done")