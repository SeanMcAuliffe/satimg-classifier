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
from keras import regularizers

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.load_data import load_datasets_reg
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

    name = input(">>> Enter experiment name: ")
    if name == "":
        name = "default"

    EXPERIMENT_NAME = name
    DESCRIPTION = "Former glory model, 512, Normalized, 5400 set"

    print("Creating model ...")

    X_train, Y_train, x_test, y_test, tr_names, val_names = load_datasets_reg(5008, downscale_dimension=512, normalize=True)


    model = keras.Sequential()
    initializer = HeUniform(seed=random.randint(1, 100))

#     model.add(keras.layers.Conv2D(filters=4,
#                                     kernel_size=(3, 3),
#                                     padding='same',
#                                     activation='relu',
#                                     kernel_initializer=initializer,
#                                     strides=(2, 2),
#                                     input_shape=(X_train.shape[1], X_train.shape[1], 1),
#                                     kernel_regularizer=keras.regularizers.l2(0.01)))

#    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#     model.add(keras.layers.Flatten())
#     #model.add(keras.layers.Dropout(0.01))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(8, activation='relu'))
#     #model.add(keras.layers.Dropout(0.01))

#     # 1 output neuron for binary classification.
#     model.add(keras.layers.Dense(1, activation='sigmoid'))

    # --------------------------------------------------------------------------
    # Input layer: 512x512 single-color band image
    # model.add(keras.layers.Conv2D(4, (3, 3), kernel_regularizer=regularizers.L1L2(), activation='relu', input_shape=(256, 256, 1)))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # model.add(keras.layers.Conv2D(8, (3, 3), kernel_regularizer=regularizers.L1L2(), activation='relu'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # model.add(keras.layers.Conv2D(16, (3, 3), kernel_regularizer=regularizers.L1L2(), activation='relu'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # model.add(keras.layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.L1L2(), activation='relu'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # model.add(keras.layers.Flatten())

    # model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L1L2()))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.5))

    # model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L1L2()))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.5))

    # # Output layer: Single continuous value representing mineral richness
    # model.add(keras.layers.Dense(1, activation='sigmoid'))

    # model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

    model.add(keras.layers.Conv2D(filters=8,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizers.L1L2(),
                                    input_shape=(X_train.shape[1], X_train.shape[1], 1)))
    #256
    model.add(keras.layers.Conv2D(filters=16,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    activation='relu',
                                    strides=(2, 2),
                                    kernel_regularizer=regularizers.L1L2(),
                                    kernel_initializer=initializer))
    #128
    model.add(keras.layers.Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_regularizer=regularizers.L1L2(),
                                    kernel_initializer=initializer))
    #64
    model.add(keras.layers.Conv2D(filters=64,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_regularizer=regularizers.L1L2(),
                                    kernel_initializer=initializer))
    #32
    model.add(keras.layers.Conv2D(filters=128,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_regularizer=regularizers.L1L2(),
                                    kernel_initializer=initializer))
    #16
    model.add(keras.layers.Conv2D(filters=256,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_regularizer=regularizers.L1L2(),
                                    kernel_initializer=initializer))
   #8
    model.add(keras.layers.Conv2D(filters=512,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                padding='same',
                                activation='relu',
                                kernel_regularizer=regularizers.L1L2(),
                                kernel_initializer=initializer))
    #4
    model.add(keras.layers.Conv2D(filters=512,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                kernel_regularizer=regularizers.L1L2(),
                                kernel_initializer=initializer))
    #4
    model.add(keras.layers.Conv2D(filters=512,
                                kernel_size=(4, 4),
                                activation='relu',
                                kernel_regularizer=regularizers.L1L2(),
                                kernel_initializer=initializer))

   # Flatten and put a fully connected layer.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))

    # Dense layers.
    model.add(keras.layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.L1()))
    model.add(keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.L1()))
    model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.L1()))
    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L1()))
    model.add(keras.layers.Dropout(0.5))

    # --------------------------------------------------------------------------
    
    model.summary()


    # Compile the model
    print("Compiling model. . .")
    opt = keras.optimizers.Adam(learning_rate=0.000001)
    model.compile(optimizer=opt, loss="mean_squared_error", metrics=['mean_absolute_error'])

    callbacks = [StopAtValAcc(0.85, 10)]

    # Run CNN.
    print("Fitting model. . .")
    history = model.fit(X_train, Y_train, verbose=1, epochs=1000, batch_size=32, validation_data=(x_test, y_test))#, callbacks=callbacks)

    save_model(model, EXPERIMENT_NAME)

    del model
    gc.collect()
    K.clear_session()

    plt.plot(history.history['mean_absolute_error'], label='Training Error')
    plt.plot(history.history['val_mean_absolute_error'], label = 'Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.ylim([0.0, 0.5])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.title(f"{name}: Error")
    #Save the plot to disc
    plt.show()
    
    #log_experiment(history, EXPERIMENT_NAME, DESCRIPTION, x_test, y_test, Y_train, tr_names, val_names)
    

if __name__ == "__main__":
    main()

