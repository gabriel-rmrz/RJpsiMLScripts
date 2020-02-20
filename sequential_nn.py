import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

f = np.load("data/featuresData.npz")
inputData = f["arr_0"]
nn_inputFeatures = inputData[0:10,:]
triggersFlags = (inputData[10:13,:]).astype(int) == 1
channelsFlags = (inputData[13:16,:]).astype(int) == 1

signal =  nn_inputFeatures [:,channelsFlags[0, :]]
normalization =  nn_inputFeatures [:,channelsFlags[1, :]]
print("Signal shape", signal.shape)
print("Normalization shape",normalization.shape)

# Spliting the samples into the training and test samples
signal_train, signal_test, normalization_train, normalization_test = train_test_split(np.transpose(signal[:,0:2500]), np.transpose(normalization[:,0:2500]), test_size = 0.3, random_state=15)

print("Signal teest shape", signal_test.shape)
print("Normalization test shape",normalization_test.shape)

batch_size = 10
n_epochs = 100
dropoutRate = 0.1

print("Defining the model.")
model = Sequential()
model.add(Dense(10, input_shape =(10,), activation='tanh'))
model.add(Dense(9, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(7, activation='tanh'))
model.add(Dense(10, activation='sigmoid'))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics = ['accuracy'])

model.fit(signal_train, normalization_train, epochs=n_epochs, verbose=1)
results = model.evaluate(signal_test, normalization_test)

print("The accuracy score on the training ser is: \t{:0.3f}".format(results[1]))
