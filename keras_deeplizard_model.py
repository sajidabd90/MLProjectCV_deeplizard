#importing required modules

import keras_deeplizard_data as data_file
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import sparse_categorical_crossentropy

#building our model

model = Sequential([
    Dense(units=16,input_shape=(1,),activation='relu'),
    Dense(units=32,activation='relu'),
    Dense(units=2,activation='softmax')
])

#compiling our model to prepare it for training

model.compile(
    optimizer=Adam(learning_rate = 0.001),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)