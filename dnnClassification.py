
import os
import numpy as np
import pandas as pd
import root_pandas

from root_pandas import read_root
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv1D, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN


inputFile = 'results-nodes_100_80_60_40_20.root'
input_df = read_root(inputFile, 'tree')
#nn_inputFeatures = shuffle(nn_inputFeatures_tmp, random_state=0)
inputFeaturesNames = ['bc_pt_predicted', 'nn_q2', 'nn_missMass2', 'nn_missPt', 'nn_varPt']
nn_inputFeatures = input_df[inputFeaturesNames]#.to_numpy()
#nn_inputFeatures = np.swapaxes(nn_inputFeatures, 1, 0)
nn_target = input_df['signalDecayPresent']

batch_size = 500
n_epochs = 100
dropoutRate = 0.2

inputLayer = Input(shape=(5,))
x = BatchNormalization()(inputLayer)
x = Dense(30, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
x = Dense(30, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
x = Dense(30, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
x = Dense(10, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
outputLayer = Dense(1, activation = 'sigmoid')(x)


model = Model(inputs = inputLayer, outputs = outputLayer)
model.compile(loss='binary_crossentropy', optimizer ='adam', metrics = ['accuracy'])
model.summary()

history = model.fit(nn_inputFeatures, 
        nn_target,
        epochs = n_epochs, 
        batch_size = batch_size, 
        verbose = 1,
        validation_split = 0.3#,
        #callbacks = [
        #    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        #    ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience=2, verbose = 1),
        #    TerminateOnNaN()]
        )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

