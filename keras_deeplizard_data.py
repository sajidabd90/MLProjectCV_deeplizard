#Importing libraries 

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler  

#creating variables for samples and labels 

train_samples = []
train_labels = []

#Generating dummy data 
# 5% outliers from each group
for i in range(50):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

#Normal data set for each group

for i in range(1000):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

#Convert them to numpy arrays 
train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_labels, train_samples = shuffle(train_labels, train_samples)

#Scale data from 0 to 1
scaler = MinMaxScaler(feature_range= (0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

