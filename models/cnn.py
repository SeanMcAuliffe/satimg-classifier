import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Load images.  
image_path = "../data/images_norm/"

print("Loading images...")

# Load labels.
labels_path = "../data/labels/labels_binary_minerals.csv"
labels_rows = pd.read_csv(labels_path, header=0, low_memory=False)
labels = labels_rows.values.tolist()
labels_dict = {}
for label in labels:
     labels_dict[label[0]] = label[1]

# Downscale images.
print("Splitting images into sets...")

X_train = []
y_train = []
X_test = []
y_test = []


for i, imagename in enumerate(os.listdir(image_path)):
   im = np.array(Image.open(os.path.join(image_path, imagename)))
   # Normalize all of the pixels values to be between 0 and 1.
   # im = im / 255.0

   if im is not None:
      if i < 3500:
         X_train.append(im)
         y_train.append(0)
      elif i < 3500 + 8000:
         X_test.append(im)
         y_test.append(labels_dict[f"{imagename[:-6]}MTL"])
      else:
         break

print("Finished splitting images.")

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


model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=4,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu',
                              input_shape=(512, 512, 1)))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

model.add(keras.layers.Conv2D(filters=4,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

model.add(keras.layers.Conv2D(filters=4,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

model.add(keras.layers.Conv2D(filters=4,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

model.add(keras.layers.Conv2D(filters=4,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

model.add(keras.layers.Conv2D(filters=4,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

model.add(keras.layers.Conv2D(filters=4,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

# Flatten and put a fully connected layer.
model.add(keras.layers.Flatten())

# Hidden layers.
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))

# 1 output neuron for binary classification.
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

# Compile the model
print("Compiling model. . .")
# Sparce cross entropy loss 
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

# Run CNN.
print("Fitting model. . .")
history = model.fit(X_train, y_train, verbose=1, epochs=5, batch_size=32, validation_data=(X_test, y_test))

################################################################
# Evaluation:

# Score the model
print("Evaluating model. . .")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy: ", test_acc)
print("Test loss: ", test_loss)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#Save the plot to disc
plt.savefig(os.path.join("../vis", "accuracy.png"))

