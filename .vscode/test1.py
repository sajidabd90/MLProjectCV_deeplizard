# Building a boilerplate for a Keras Sequential Model
from keras.models import Sequential
from keras.layers import Dense, Activation


layers = [
    Dense(units = 3, input_shape=(2,), activation='relu',),
    Dense(units =2 , activation= 'softmax')
]