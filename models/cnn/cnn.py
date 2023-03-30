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

# Load images.  
image_path = "./data/images/B7_512x512/1_batch_unrolled"

print("Loading images...")

description = """ Here is the experiment I intend to run """

# Load labels.
labels_path = "./data/labels/B7_512x512/labels_binary_minerals.csv"
labels_rows = pd.read_csv(labels_path, header=0, low_memory=False)
labels = labels_rows.values.tolist()
labels_dict = {}
for label in labels:
     labels_dict[label[0]] = label[1]

num_pixels = 262144 # 512 x 512

# Downscale images.
print("Downscaling images...")

X_train = []
y_train = []
X_test = []
y_test = []

for i, imagename in enumerate(os.listdir(image_path)):
   #subprocess.run(["convert", os.path.join(image_path, imagename), "-resize!", "512x512", os.path.join(image_path, imagename)])
   im = np.array(Image.open(os.path.join(image_path, imagename)))
   im = im.flatten()
   im_pixels = len(im)
   difference = num_pixels - im_pixels

   if difference > 0:
      zeros = np.zeros((difference, 1), dtype=np.uint8)
      im = np.append(im, zeros)

   im = im.reshape((512, 512))

   # m = skio.imread(os.path.join(image_path, imagename), plugin="pil")
   # image_resized = resize(im, (512, 512), anti_aliasing=True)

   if im is not None:
      if i < 4000:
         X_train.append(im)
         y_train.append(labels_dict[f"{imagename[:-6]}MTL"])
      else:
         X_test.append(im)
         y_test.append(labels_dict[f"{imagename[:-6]}MTL"])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

# Convert images to inputs of format expected by Tensorflow.
# https://www.tensorflow.org/tutorials/images/cnn

# Create training / testing sets.

# Create the model
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

print("Creating model. . .")

################################################################
# MODEL 1:

# model = keras.Sequential()
# model.add(keras.layers.Conv2D(filters=8,
#                               kernel_size=(16, 16),
#                               strides=(8, 8),
#                               padding='same',
#                               activation='relu',
#                               input_shape=(512, 512, 1)))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# model.add(keras.layers.Conv2D(filters=4,
#                               kernel_size=(8, 8),
#                               strides=(2, 2),
#                               input_shape=(64, 64, 8)))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# model.add(keras.layers.Flatten())

# model.add(keras.layers.Dense(64, activation='sigmoid'))
# model.add(keras.layers.Dense(1, activation='softmax'))
# model.summary()

# model = keras.Sequential()
# model.add(keras.layers.Conv2D(filters=1,
#                               kernel_size=(16, 16),
#                               strides=(8, 8),
#                               padding='same',
#                               activation='relu',
#                               input_shape=(512, 512, 1)))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(keras.layers.MaxPooling2D((2, 2)))
# model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(keras.layers.MaxPooling2D((2, 2)))
# model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dense(2))
# model.summary()

# # Compile the model
# print("Compiling model. . .")
# # Sparce cross entropy loss 
# # https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
# model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# # Run CNN.
# print("Fitting model. . .")
# history = model.fit(X_train, y_train, verbose=1, epochs=15, batch_size=32, validation_data=(X_test, y_test))

################################################################
# MODEL 2:

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=8,
                              kernel_size=(16, 16),
                              strides=(4, 4),
                              padding='same',
                              activation='relu',
                              input_shape=(512, 512, 1)))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

model.add(keras.layers.Conv2D(filters=4,
                              kernel_size=(8, 8),
                              strides=(2, 2),
                              activation='relu',
                              input_shape=(128, 128, 16)))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))
model.summary()

# # Compile the model
# print("Compiling model. . .")
# # Sparce cross entropy loss 
# # https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
# model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# # Run CNN.
# print("Fitting model. . .")
# history = model.fit(X_train, y_train, verbose=1, epochs=15, batch_size=32, validation_data=(X_test, y_test))

################################################################
# MODEL 3:

# model = keras.Sequential()
# model.add(keras.layers.Conv2D(filters=8,
#                               kernel_size=(3, 3),
#                               padding='same',
#                               activation='relu',
#                               input_shape=(512, 512, 1)))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# model.add(keras.layers.Conv2D(filters=16,
#                               kernel_size=(3, 3),
#                               padding='same',
#                               activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# model.add(keras.layers.Conv2D(filters=32,
#                               kernel_size=(3, 3),
#                               padding='same',
#                               activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# model.add(keras.layers.Conv2D(filters=64,
#                               kernel_size=(3, 3),
#                               padding='same',
#                               activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# model.add(keras.layers.Conv2D(filters=64,
#                               kernel_size=(3, 3),
#                               padding='same',
#                               activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# model.add(keras.layers.Conv2D(filters=64,
#                               kernel_size=(3, 3),
#                               padding='same',
#                               activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# model.add(keras.layers.Conv2D(filters=64,
#                               kernel_size=(3, 3),
#                               padding='same',
#                               activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                     strides=(2, 2),
#                                     padding='same'))

# # Flatten and put a fully connected layer.
# model.add(keras.layers.Flatten())

# # Hidden layers.
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dense(256, activation='relu'))

# # 1 output neuron for binary classification.
# model.add(keras.layers.Dense(1, activation='sigmoid'))
# model.summary()

# Compile the model
print("Compiling model. . .")
# Sparce cross entropy loss 
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

# Run CNN.
print("Fitting model. . .")
history = model.fit(X_train, y_train, verbose=1, epochs=10, batch_size=16, validation_data=(X_test, y_test))

################################################################
# Evaluation:

# Score the model
print("Evaluating model. . .")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy: ", test_acc)
print("Test loss: ", test_loss)
print(model.summary())

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#Save the plot to disc
plt.savefig(os.path.join("vis", "accuracy.png"))
