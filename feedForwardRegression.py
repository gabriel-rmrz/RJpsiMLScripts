import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from random import randint
seed = 7
np.random.seed(seed)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

#%tensorflow_version 2.x
#%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)

########################################################################
########################################################################
# Preparation of the input featues
########################################################################
########################################################################

inputFile="data/featuresData.npz"
f = np.load(inputFile)#, allow_pickle=True)
inputData = f["arr_0"]
inputData=np.swapaxes(inputData, 0, 1)
print(inputData.shape)

inputData = inputData[inputData[:,23].astype(int) == 1,] # 22 for signal channel and 23 for normalization channel
target=inputData[:,0:4]
print (target.shape)

reg_input=inputData[:,[26, 27, 28, 16, 17, 18, 30, 31, 32, 45, 46, 47, 51, 52, 53, 71, 72, 73, 74, 75, 76, 77, 78, 79]]
print (reg_input.shape)

batch_size = 100
validation_frac = 0.2
n_epochs = 500
dropoutRate = 0.1
inputData.shape

from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(reg_input)
import copy

t_data=reg_input[0:int((1-validation_frac)*len(reg_input))]
v_data=reg_input[int((1-validation_frac)*len(reg_input)):len(reg_input)]

t_target=target[0:int((1-validation_frac)*len(target))]
v_target=target[int((1-validation_frac)*len(target)):len(target)]

scaled_t_data=sc.transform(t_data)
scaled_v_data=sc.transform(v_data)
#scaled_t_data=copy.copy(t_data)
#scaled_v_data=copy.copy(v_data)
#scaled_t_data=sc.transform(scaled_t_data)
#scaled_v_data=sc.transform(scaled_v_data)

data_len=scaled_t_data.shape[1]

########################################################################
########################################################################
# Model definition
########################################################################
########################################################################

model = Sequential()

model.add(Dense(21, activation="relu", kernel_initializer="glorot_uniform", input_dim=data_len))
model.add(Dropout(dropoutRate))
model.add(Dense(100, activation="relu", kernel_initializer="glorot_uniform"))
model.add(Dropout(dropoutRate))
model.add(Dense(50, activation="relu", kernel_initializer="glorot_uniform"))
model.add(Dropout(dropoutRate))
model.add(Dense(20, activation="relu", kernel_initializer="glorot_uniform"))
model.add(Dropout(dropoutRate))
model.add(Dense(4, kernel_initializer="glorot_uniform"))
print('compiling')


model.compile(loss='mse', optimizer='adam',metrics=['mse', 'mae'])

model.summary()


history=model.fit(scaled_t_data , 
                  t_target, 
                  nb_epoch=n_epochs, verbose=1,batch_size=batch_size, 
                  validation_data=(scaled_v_data , v_target))

plt.figure(num=None, figsize=(3, 3), dpi = 300, facecolor='w', edgecolor='k') 
plotsDir = 'plots/dnnFeedForward/'
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model mse')
plt.yscale('log')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mae')
plt.yscale('log')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

prediction_p4 = model.predict(scaled_v_data)
prediction_px = prediction_p4[:,0]
prediction_py = prediction_p4[:,1]
prediction_pz = prediction_p4[:,2]
prediction_E = prediction_p4[:,3]

print(prediction_px.shape, "   ",scaled_v_data[:,0].shape)


x = np.linspace(-60., 60., 1000)
plt.figure(num=None, figsize=(6, 6), dpi = 300, facecolor='w', edgecolor='k') 
plt.scatter(v_data[:,0], prediction_px, c='tab:red', alpha = 0.2, label = 'corrected_data') 
plt.scatter(v_data[:,0],v_target[:,0],  c='tab:blue', alpha = 0.2, label = 'target_data') 
plt.plot(x, x + 0, '-g', label= 'input_data')
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True) 
#plt.savefig('test.png')


x = np.linspace(-60., 60., 1000)
plt.scatter(v_data[:,1], prediction_py, c='tab:red', alpha = 0.2, label = 'corrected_data')
plt.scatter(v_data[:,1],v_target[:,1],  c='tab:blue', alpha = 0.2, label = 'target_data')
plt.plot(x, x + 0, '-g', label= 'input_data')
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True)
#plt.savefig('test.png')

x = np.linspace(-100., 100., 1000)
plt.scatter(v_data[:,2], prediction_pz, c='tab:red', alpha = 0.2, label = 'corrected_data')
plt.scatter(v_data[:,2],v_target[:,2],  c='tab:blue', alpha = 0.2, label = 'target_data')
plt.plot(x, x + 0, '-g', label= 'input_data')
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True)
#plt.savefig('test.png')

x = np.linspace(0., 100., 1000)
plt.scatter(v_data[:,3], prediction_E, c='tab:red', alpha = 0.2, label = 'corrected_data')
plt.scatter(v_data[:,3],v_target[:,3],  c='tab:blue', alpha = 0.2, label = 'target_data')
plt.plot(x, x + 0, '-g', label= 'input_data')
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True)
#plt.savefig('test.png')

plt.clf()
minVal = 0
maxVal = 60
binwidth = 4
bins=range(minVal, maxVal + binwidth, binwidth)
plt.hist(v_data[:,3], bins=bins, color='tab:green', label='input_data', density=True)
plt.hist(prediction_E, bins=bins, color='tab:red', label='corrected_data', density=True)
plt.hist(v_target[:,3], bins=bins,  color='tab:blue', label='target_data', density=True)
plt.legend(loc='upper right', fontsize='x-small')
#plt.xlim(-50,50)
#plt.grid(True)
plt.savefig(plotsDir+'hEnergyComparison.pdf')
