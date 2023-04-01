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
from keras import backend as K
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.load_data import load_datasets
from utils.test_acc_callback import StopAtTrainAcc, StopAtValAcc
from utils.utils import log_experiment, save_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    except RuntimeError as e:
        print(e)


def main():

    EXPERIMENT_NAME = "flatius_extremius"
    DESCRIPTION = "Normalizing the images, 512x512, 5454 images, generic architecture"

    X_train, Y_train, x_test, y_test, tr_names, val_names = load_datasets(5454, downscale_dimension=512, normalize=True)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", Y_train.shape)
    print("X_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    print("Creating model ...")

    model = keras.Sequential()

    initializer = HeUniform(seed=random.randint(1, 100))

    # Generic ------------------------------------------------------------------
    # model.add(keras.layers.Conv2D(filters=8,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 strides=(2, 2),
    #                                 input_shape=(X_train.shape[1], X_train.shape[1], 1)))

    # model.add(keras.layers.Conv2D(filters=16,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 strides=(2, 2),
    #                                 kernel_initializer=initializer))

    # model.add(keras.layers.Conv2D(filters=32,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer))

    # # Flatten and put a fully connected layer.
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())

    # # Dense layers.
    # model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dense(32, activation='relu'))

    # MAX POOLING MODEL: -------------------------------------------------------

    # --------------------------------------------------------------------------

    # Paramus Minimus ----------------------------------------------------------

    # model.add(keras.layers.Dense(3, activation='relu'))
    # --------------------------------------------------------------------------

    # ULTRA_WIDE
    # Flatius Extremius
    # model.add(keras.layers.Conv2D(filters=2,
    #                                 kernel_size=(4, 4),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 strides=(3, 3),
    #                                 input_shape=(X_train.shape[1], X_train.shape[1], 1)))

    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(6, activation='relu'))

    # Paramus Flatius(topius)
    # model.add(keras.layers.Conv2D(filters=4,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 strides=(2, 2),
    #                                 input_shape=(X_train.shape[1], X_train.shape[1], 1)))

    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(8, activation='relu'))

    # Wide-Conv, Wide-Dense ----------------------------------------------------
    # Paramus Squatius
    # model.add(keras.layers.Conv2D(filters=4,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 strides=(2, 2),
    #                                 input_shape=(X_train.shape[1], X_train.shape[1], 1)))

    # model.add(keras.layers.Conv2D(filters=8,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 strides=(2, 2),
    #                                 kernel_initializer=initializer))

    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())
    
    # model.add(keras.layers.Dense(32, activation='relu'))
    # model.add(keras.layers.Dense(16, activation='relu'))
    # model.summary()
    # quit()
    # --------------------------------------------------------------------------

    # Wide-Conv, Medium-Dense --------------------------------------------------
    # --------------------------------------------------------------------------

    # Wide-Conv, Narrow-Dense --------------------------------------------------
    # --------------------------------------------------------------------------

    # Medium-Conv, Wide-Dense --------------------------------------------------
    # --------------------------------------------------------------------------

    # Medium-Conv, Medium-Dense ------------------------------------------------
    # Paramus Medius
    # model.add(keras.layers.Conv2D(filters=8,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 strides=(2, 2),
    #                                 input_shape=(X_train.shape[1], X_train.shape[1], 1)))

    # # Output 128
    # model.add(keras.layers.Conv2D(filters=16,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 strides=(2, 2),
    #                                 kernel_initializer=initializer))

    # # Output: 64x64
    # model.add(keras.layers.Conv2D(filters=32,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer))
    # # output 32
    # model.add(keras.layers.Conv2D(filters=64,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer))
    
    # # Output 16
    # model.add(keras.layers.Conv2D(filters=128,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer))
    

    # model.add(keras.layers.Conv2D(filters=256,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer))

    # # Flatten and put a fully connected layer.
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())

    # # Dense layers.
    # model.add(keras.layers.Dense(1024, activation='relu'))
    # model.add(keras.layers.Dense(512, activation='relu'))
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dense(32, activation='relu'))
    # --------------------------------------------------------------------------

    # Medium-Conv, Narrow-Dense ------------------------------------------------
    # --------------------------------------------------------------------------

    # Narrow-Conv, Wide-Dense --------------------------------------------------
    # --------------------------------------------------------------------------

    # Narrow-Conv, Medium-Dense ------------------------------------------------
    # --------------------------------------------------------------------------

    # Narrow-Conv, Narrow-Dense ------------------------------------------------
    # Paramus Telescopium
    # Output: 256x256
    model.add(keras.layers.Conv2D(filters=8,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    strides=(2, 2),
                                    input_shape=(X_train.shape[1], X_train.shape[1], 1)))

    # Output 128
    model.add(keras.layers.Conv2D(filters=16,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    activation='relu',
                                    strides=(2, 2),
                                    kernel_initializer=initializer))

    # Output: 64x64
    model.add(keras.layers.Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer))
    # output 32
    model.add(keras.layers.Conv2D(filters=64,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer))
    
    # Output 16
    model.add(keras.layers.Conv2D(filters=128,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer))

    # Output: 8x8
    model.add(keras.layers.Conv2D(filters=256,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer))
    # Output 4x4
    model.add(keras.layers.Conv2D(filters=512,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer))
    # # output 2x2
    # model.add(keras.layers.Conv2D(filters=1024,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer))
    # # output 1x1
    # model.add(keras.layers.Conv2D(filters=2048,
    #                                 kernel_size=(2, 2),
    #                                 activation='relu',
    #                                 kernel_initializer=initializer))
    
    # Flatten and put a fully connected layer.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.BatchNormalization())

    # # Dense layers.
    # model.add(keras.layers.Dense(8192, activation='relu'))
    # model.add(keras.layers.Dense(4096, activation='relu'))
    # model.add(keras.layers.Dense(2048, activation='relu'))
    # model.add(keras.layers.Dense(1024, activation='relu'))
    # model.add(keras.layers.Dense(512, activation='relu'))
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dense(32, activation='relu'))
    # --------------------------------------------------------------------------


    # 1 output neuron for binary classification.
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    # Compile the model
    print("Compiling model. . .")
    opt = keras.optimizers.Adam(learning_rate=0.0000001)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

    callbacks = [StopAtTrainAcc(0.95, 20)]

    # Run CNN.
    print("Fitting model. . .")
    history = model.fit(X_train, Y_train, verbose=1, epochs=1000, batch_size=32, validation_data=(x_test, y_test), callbacks=callbacks)

    save_model(model, EXPERIMENT_NAME)

    del model
    gc.collect()
    K.clear_session()
    
    log_experiment(history, EXPERIMENT_NAME, DESCRIPTION, x_test, y_test, Y_train, tr_names, val_names)
    

if __name__ == "__main__":
    main()

