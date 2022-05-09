import pickle

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model

import pickle

width = 1280
height = 720

eye_data = []
calibration_points = []
with open("eye_data", "rb") as ed:
    eye_data = np.array(pickle.load(ed))
with open("calibration_points", "rb") as cp:
    calibration_points = np.array(pickle.load(cp))
on_points = np.array([[x[0]*width+width/2, x[1]*height+height/2] for x in calibration_points])

def regressor_model():
    input_x = Input(shape = (480,))
    x = Dense(40, activation = 'relu')(input_x)
    x = Dense(40, activation = 'relu')(x)
    x = Dense(40, activation = 'relu')(x)
    x = Dense(40, activation = 'relu')(x)
    output = Dense(2)(x)
    model = Model(inputs = input_x, outputs = output)
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    return model


# Training

model = regressor_model()
checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint', monitor="loss", mode="min",
                                                save_best_only=True, verbose=0)
callbacks = [checkpoint]

eye_data= np.reshape(eye_data,(eye_data.shape[0], 480))

log_train = model.fit(eye_data, calibration_points, epochs = 1000,  verbose = 1)
score = model.evaluate(eye_data, calibration_points, verbose = 0)
print('Test loss:', score)