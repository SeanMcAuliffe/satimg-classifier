import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras.initializers import HeUniform, HeNormal
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.load_data import load_datasets

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    except RuntimeError as e:
        print(e)

def main():
   X_train, Y_train, x_test, y_test = load_datasets(10000, downscale_dimension=128, normalize=True)

   print("X_train shape: ", X_train.shape)
   print("y_train shape: ", Y_train.shape)
   print("X_test shape: ", x_test.shape)
   print("y_test shape: ", y_test.shape)

   print("Creating model ...")

   model = keras.Sequential()

   initializer = HeUniform(seed=random.randint(1, 100))

   model.add(keras.layers.Conv2D(filters=8,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 kernel_initializer=initializer,
                                 strides=(2, 2),
                                 input_shape=(X_train.shape[1], X_train.shape[1], 1)))

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
   opt = keras.optimizers.Adam(learning_rate=0.0000001)
   model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

   # Run CNN.
   print("Fitting model. . .")
   history = model.fit(X_train, Y_train, verbose=1, epochs=100, batch_size=32, validation_data=(x_test, y_test))

   ################################################################
   # Evaluation:

   # Score the model
   print("Evaluating model. . .")
   test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
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


if __name__ == "__main__":
    main()

