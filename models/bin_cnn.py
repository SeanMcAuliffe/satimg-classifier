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

    name = input(">>> Enter experiment name: ")
    if name == "":
        name = "default"

    EXPERIMENT_NAME = name
    DESCRIPTION = "Former glory model, 512, Normalized, 5400 set"

    print("Creating model ...")

    model = keras.Sequential()
    initializer = HeUniform(seed=random.randint(1, 100))
    
    # --------------------------------------------------------------------------
    # C-D
    # M-M = 0.7148148417472839
    # M-S = 0.7268518805503845
    # Glorius = 0.7046296000480652
    # S-S = 0.7250000238418579
    # L-S = 0.7138888835906982
    # Flat verification (S-S) = 0.7129629850387573
    # Small small 16 filters: 0.7157407402992249
    # --------------------------------------------------------------------------

    # Generic ------------------------------------------------------------------
    # model.add(keras.layers.Conv2D(filters=8,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 strides=(3, 3),
    #                                 input_shape=(X_train.shape[1], X_train.shape[1], 1),
    #                                 kernel_regularizer=keras.regularizers.l2(0.01)))

    # model.add(keras.layers.Conv2D(filters=16,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 strides=(2, 2),
    #                                 kernel_initializer=initializer,
    #                                 kernel_regularizer=keras.regularizers.l2(0.01)))

    # model.add(keras.layers.Conv2D(filters=32,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 kernel_regularizer=keras.regularizers.l2(0.01)))
    
    # model.add(keras.layers.Conv2D(filters=64,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 strides=(3, 3),
    #                                 kernel_regularizer=keras.regularizers.l2(0.01)))

    # model.add(keras.layers.Conv2D(filters=128,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 strides=(2, 2),
    #                                 kernel_initializer=initializer,
    #                                 kernel_regularizer=keras.regularizers.l2(0.01)))

    # model.add(keras.layers.Conv2D(filters=512,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 kernel_regularizer=keras.regularizers.l2(0.01)))
    
    # # Flatten and put a fully connected layer.
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())

    # # Dense layers.
    # model.add(keras.layers.Dense(128, activation='relu'))
    # # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dense(32, activation='relu'))


    # Flatius Extremius --------------------------------------------------------
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
    # --------------------------------------------------------------------------

    # Paramus Flatius(topius) --------------------------------------------------
    model.add(keras.layers.Conv2D(filters=16,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    strides=(2, 2),
                                    input_shape=(256, 256, 1),
                                    kernel_regularizer=keras.regularizers.l2(0.01)))

    # model.add(keras.layers.Conv2D(filters=16,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 strides=(2, 2),
    #                                 kernel_initializer=initializer,
    #                                 kernel_regularizer=keras.regularizers.l2(0.01)))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.01))
    model.summary()
    X_train, Y_train, x_test, y_test, tr_names, val_names = load_datasets(5008, downscale_dimension=256, normalize=True)
    # --------------------------------------------------------------------------

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
    # model.add(keras.layers.Conv2D(filters=8,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer,
    #                                 strides=(2, 2),
    #                                 input_shape=(X_train.shape[1], X_train.shape[1], 1),
    #                                 activity_regularizer=keras.regularizers.l2(0.01),
    #                                 kernel_regularizer=keras.regularizers.l2(0.01)))
    
    # model.add(keras.layers.BatchNormalization())

    # # Output 128
    # model.add(keras.layers.Conv2D(filters=16,
    #                                 kernel_size=(3, 3),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 strides=(2, 2),
    #                                 kernel_initializer=initializer),
    #                                 activity_regularizer=keras.regularizers.l2(0.01),
    #                                 kernel_regularizer=keras.regularizers.l2(0.01))
    
    # model.add(keras.layers.BatchNormalization())

    # # Output: 64x64
    # model.add(keras.layers.Conv2D(filters=32,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer),
    #                                 activity_regularizer=keras.regularizers.l2(0.01),
    #                                 kernel_regularizer=keras.regularizers.l2(0.01))
    
    # model.add(keras.layers.BatchNormalization())
    # # output 32
    # model.add(keras.layers.Conv2D(filters=64,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer),
    #                                 activity_regularizer=keras.regularizers.l2(0.01),
    #                                 kernel_regularizer=keras.regularizers.l2(0.01))
    
    # model.add(keras.layers.BatchNormalization())
    
    # # Output 16
    # model.add(keras.layers.Conv2D(filters=128,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer),
    #                                 activity_regularizer=keras.regularizers.l2(0.01),
    #                                 kernel_regularizer=keras.regularizers.l2(0.01))
    
    # model.add(keras.layers.BatchNormalization())

    # # Output: 8x8
    # model.add(keras.layers.Conv2D(filters=256,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer),
    #                                 activity_regularizer=keras.regularizers.l2(0.01),
    #                                 kernel_regularizer=keras.regularizers.l2(0.01))
    
    # model.add(keras.layers.BatchNormalization())
    # # Output 4x4
    # model.add(keras.layers.Conv2D(filters=512,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding='same',
    #                                 activation='relu',
    #                                 kernel_initializer=initializer),
    #                                 activity_regularizer=keras.regularizers.l2(0.01),
    #                                 kernel_regularizer=keras.regularizers.l2(0.01))
    
    # model.add(keras.layers.BatchNormalization())

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
    
    # # Flatten and put a fully connected layer.
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())

    # # # Dense layers.
    # model.add(keras.layers.Dense(8192, activation='relu'))
    # model.add(keras.layers.Dense(4096, activation='relu'))
    # model.add(keras.layers.Dense(2048, activation='relu'))
    # model.add(keras.layers.Dense(1024, activation='relu'))
    # model.add(keras.layers.Dense(512, activation='relu'))
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dense(32, activation='relu'))
    # --------------------------------------------------------------------------

    # Former glory model -------------------------------------------------------
    #512
#     model.add(keras.layers.Conv2D(filters=8,
#                                     kernel_size=(3, 3),
#                                     strides=(2, 2),
#                                     padding='same',
#                                     activation='relu',
#                                     kernel_initializer=initializer,
#                                     kernel_regularizer=regularizers.L1L2(),
#                                     input_shape=(X_train.shape[1], X_train.shape[1], 1)))
#     #256
#     model.add(keras.layers.Conv2D(filters=16,
#                                     kernel_size=(3, 3),
#                                     padding='same',
#                                     activation='relu',
#                                     strides=(2, 2),
#                                     kernel_regularizer=regularizers.L1L2(),
#                                     kernel_initializer=initializer))
#     #128
#     model.add(keras.layers.Conv2D(filters=32,
#                                     kernel_size=(3, 3),
#                                     strides=(2, 2),
#                                     padding='same',
#                                     activation='relu',
#                                     kernel_regularizer=regularizers.L1L2(),
#                                     kernel_initializer=initializer))
#     #64
#     model.add(keras.layers.Conv2D(filters=64,
#                                     kernel_size=(3, 3),
#                                     strides=(2, 2),
#                                     padding='same',
#                                     activation='relu',
#                                     kernel_regularizer=regularizers.L1L2(),
#                                     kernel_initializer=initializer))
#     #32
#     model.add(keras.layers.Conv2D(filters=128,
#                                     kernel_size=(3, 3),
#                                     strides=(2, 2),
#                                     padding='same',
#                                     activation='relu',
#                                     kernel_regularizer=regularizers.L1L2(),
#                                     kernel_initializer=initializer))
#     #16
#     model.add(keras.layers.Conv2D(filters=256,
#                                     kernel_size=(3, 3),
#                                     strides=(2, 2),
#                                     padding='same',
#                                     activation='relu',
#                                     kernel_regularizer=regularizers.L1L2(),
#                                     kernel_initializer=initializer))
#    #8
#     model.add(keras.layers.Conv2D(filters=512,
#                                 kernel_size=(3, 3),
#                                 strides=(2, 2),
#                                 padding='same',
#                                 activation='relu',
#                                 kernel_regularizer=regularizers.L1L2(),
#                                 kernel_initializer=initializer))
#     #4
#     model.add(keras.layers.Conv2D(filters=512,
#                                 kernel_size=(3, 3),
#                                 padding='same',
#                                 activation='relu',
#                                 kernel_regularizer=regularizers.L1L2(),
#                                 kernel_initializer=initializer))
#     #4
#     model.add(keras.layers.Conv2D(filters=512,
#                                 kernel_size=(4, 4),
#                                 activation='relu',
#                                 kernel_regularizer=regularizers.L1L2(),
#                                 kernel_initializer=initializer))

#    # Flatten and put a fully connected layer.
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dropout(0.5))

#     # Dense layers.
#     model.add(keras.layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.L1L2()))
#     model.add(keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.L1L2()))
#     model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.L1L2()))
#     model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L1L2()))
#     model.add(keras.layers.Dropout(0.5))
    # --------------------------------------------------------------------------


    # 1 output neuron for binary classification.
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # model.summary()

    # Compile the model
    print("Compiling model. . .")
    opt = keras.optimizers.Adam(learning_rate=0.000001)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

    callbacks = [StopAtValAcc(0.85, 10)]

    # Run CNN.
    print("Fitting model. . .")
    history = model.fit(X_train, Y_train, verbose=1, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=callbacks)

    save_model(model, EXPERIMENT_NAME)

    del model
    gc.collect()
    K.clear_session()
    
    log_experiment(history, EXPERIMENT_NAME, DESCRIPTION, x_test, y_test, Y_train, tr_names, val_names)
    

if __name__ == "__main__":
    main()

