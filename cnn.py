import os
import numpy as np
from sklearn.model_selection import train_test_split

f = np.load("data/featuresData.npz")
inputData = f["arr_0"]
nn_inputFeatures = inputData[0:10,:]
triggersFlags = (inputData[10:13,:]).astype(int) == 1
channelsFlags = (inputData[13:15,:]).astype(int) == 1

'''
signal =  nn_inputFeatures [:,channelsFlags[0, :]]
normalization =  nn_inputFeatures [:,channelsFlags[1, :]]

print("Signal shape", signal.shape)
print("Normalization shape",normalization.shape)

# Spliting the samples into the training and test samples
signal_train, signal_test, normalization_train, normalization_test = train_test_split(np.transpose(signal[:,0:2500]), np.transpose(normalization[:,0:2500]), test_size = 0.3, random_state=15)

print("Signal teest shape", signal_test.shape)
print("Normalization test shape",normalization_test.shape)
'''

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv1D, BatchNormalization

print("Defining the model.")
batch_size = 500
n_epochs = 500
dropoutRate = 0.25

inputLayer = Input(shape=(10,))
x = BatchNormalization()(inputLayer)
x = Dense(30, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
x = Dense(30, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
x = Dense(20, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
x = Dense(10, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
outputLayer = Dense(2, activation = 'sigmoid')(x)

model = Model(inputs = inputLayer, outputs = outputLayer)
model.compile(loss='categorical_crossentropy', optimizer ='adam')
model.summary()


print("Training the model.")
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
history = model.fit(np.transpose(nn_inputFeatures), np.transpose(channelsFlags), epochs = n_epochs, batch_size = batch_size, verbose = 2,
        validation_split = 0.3,
        callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience=2, verbose = 1),
        TerminateOnNaN()])

# Plot training history
from matplotlib import pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
