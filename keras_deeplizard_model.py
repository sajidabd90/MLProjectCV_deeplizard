#importing required modules

import keras_deeplizard_data as data_file
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import sparse_categorical_crossentropy
from numpy import argmax
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt 
from custom_plot_function import plot_confusion_matrix
from keras.models import load_model

#building our model

model = Sequential([
    Dense(units=16,input_shape=(1,),activation='relu'),
    Dense(units=32,activation='relu'),
    Dense(units=2,activation='softmax')
])

#compiling our model to prepare it for training

model.compile(
    optimizer=Adam(learning_rate = 0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#training the model 

model.fit(
          x=data_file.scaled_train_samples,
          y=data_file.train_labels,
          batch_size=10,
          validation_split=0.15,
          epochs=30,
          verbose=2
          )

#Regressional probability prediction 
predictions = model.predict(
         x=data_file.scaled_test_samples,
         batch_size=10,
         verbose=0
        )

#Classificational prediction 
rounded_predictions = argmax(predictions,axis=-1)

#creating confusion matrix 

cm = confusion_matrix(y_true= data_file.test_labels,y_pred=rounded_predictions)

plot_confusion_matrix(cm=cm,classes=['No side effects', 'Had side effects'],title='Confusion Matrix')

#saving current model
model.save('models/practice_side_effects_model.h5')