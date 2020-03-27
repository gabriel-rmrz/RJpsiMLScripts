import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import randint
import copy
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
class dataSet():
  inputFile=inputFile
  f = np.load(inputFile)#, allow_pickle=True)
  inputData = f["arr_0"]
  inputData = np.swapaxes(inputData, 0, 1)
  validation_frac = 0.3

  def __init__(self, inputFile = "featuresData.npz"): 
  #inputData_norm = inputData[inputData[:,29].astype(int) == 1,] # 28 for signal channel and 29 for normalization channel
  #inputData_norm = inputData_norm[-4001:,:] 
  #inputData_sig = inputData[inputData[:,28].astype(int) == 1,] # 28 for signal channel and 29 for normalization channel
  #inputData_sig = inputData_sig[:1000,:] 
  #inputData = np.append(inputData_norm, inputData_sig, axis=0)

  target=inputData[:,[0,3,4]]
  reg_input=inputData[:,[32, 33, 34, 35, 20, 21, 22, 23, 24, 36, 37, 38, 39, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]]
  print (reg_input.shape)

  sc = StandardScaler().fit(reg_input)
  
  t_data=reg_input[0:int((1-validation_frac)*len(reg_input))]
  v_data=reg_input[int((1-validation_frac)*len(reg_input)):len(reg_input)]
  
  t_target=target[0:int((1-validation_frac)*len(target))]
  v_target=target[int((1-validation_frac)*len(target)):len(target)]
  
  t_allData=inputData[0:int((1-validation_frac)*len(inputData))]
  v_allData=inputData[int((1-validation_frac)*len(inputData)):len(inputData)]
  
  scaled_t_data=sc.transform(t_data)
  scaled_v_data=sc.transform(v_data)
  
  def set(self):
      pass
  def getTrainingSubsample(self):
      pass
  def getValidationSubsample(self):
      pass
  def save(self):
      pass
  def setValidationFraction(self):
      pass
        
class dataSetType():
  def __init__(self, name= 'all', selectedRows=...)

  #return inputData, reg_input, target

batch_size = 50 
n_epochs = 200
dropoutRate = 0.1
inputData.shape


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
model.add(Dense(60, activation="relu", kernel_initializer="glorot_uniform"))
model.add(Dropout(dropoutRate))
model.add(Dense(40, activation="relu", kernel_initializer="glorot_uniform"))
model.add(Dropout(dropoutRate))
model.add(Dense(20, activation="relu", kernel_initializer="glorot_uniform"))
model.add(Dropout(dropoutRate))
model.add(Dense(3, kernel_initializer="glorot_uniform"))
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
plt.clf()

prediction_p4 = model.predict(scaled_v_data)
prediction_pt = prediction_p4[:,0]
prediction_pz = prediction_p4[:,1]
prediction_E = prediction_p4[:,2]

x = np.linspace(0., 60., 1000)
plt.figure(num=None, figsize=(6, 6), dpi = 300, facecolor='w', edgecolor='k') 
plt.scatter(v_target[:,0], prediction_pt, c='tab:orange', alpha = 0.2, label = 'NN prediction') 
plt.scatter(v_target[:,0], v_allData[:,5], c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
#plt.scatter(v_target[:,0], prediction_pt, c='tab:red', alpha = 0.2) 
#plt.scatter(v_target[:,0], prediction_pt, c='tab:red', alpha = 0.2, label = 'corrected_data') 
plt.plot(x, x + 0, '-b', label= 'Truth MC')
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True) 
plt.xlabel("pt(Truth MC) [GeV]")
plt.ylabel("pt [GeV]")
plt.savefig(plotsDir+'sPtComparison.png')
plt.clf()


x = np.linspace(-150., 150., 1000)
plt.scatter(v_target[:,1], prediction_pz, c='tab:orange', alpha = 0.2, label = 'NN prediction') 
plt.scatter(v_target[:,1], v_allData[:,8], c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
#plt.scatter(v_target[:,1], prediction_pz, c='tab:red', alpha = 0.2)
plt.plot(x, x + 0, '-b', label= 'Truth MC')
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True)
plt.xlabel("pz(Truth MC) [GeV]")
plt.ylabel("pz(NN prediction) [GeV]")
plt.savefig(plotsDir+'sPzComparison.png')
plt.clf()

x = np.linspace(0., 200., 1000)
plt.scatter(v_target[:,2], prediction_E, c='tab:orange', alpha = 0.2, label = 'NN prediction') 
plt.scatter(v_target[:,2], v_allData[:,9], c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
plt.plot(x, x + 0, '-b', label= 'Truth MC')
#plt.scatter(v_target[:,2], prediction_E, c='tab:blue', alpha = 0.2)
#plt.scatter(v_target[:,2], prediction_E, c='tab:red', alpha = 0.2)
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True)
plt.xlabel("Energy(Truth MC) [GeV]")
plt.ylabel("Energy(NN prediction) [GeV]")
plt.savefig(plotsDir+'sEnergyComparison.png')
plt.clf()

minVal = 0
maxVal = 60
binwidth = 4
bins=range(minVal, maxVal + binwidth, binwidth)
plt.hist(v_data[:,4], bins=bins, color='tab:gray', label='Non corrected', histtype = 'step', density=True)
plt.hist(prediction_E, bins=bins, color='tab:orange', label='NN prediction', histtype = 'step', density=True)
plt.hist(v_target[:,2], bins=bins,  color='tab:cyan', label='Jona recipy', histtype = 'step', density=True)
plt.legend(loc='upper right', fontsize='x-small')
#plt.xlim(-50,50)
#plt.grid(True)
plt.xlabel("Energy [GeV]")
plt.savefig(plotsDir+'hEnergyComparison.png')
plt.clf()




########################################################################3
v_pt_measured = v_allData[:,32]
v_pt_corrected = v_allData[:,5]
v_pt_target = v_target[:,0]
v_pt_predicted = prediction_pt


minVal = 0. 
maxVal = 2.
binwidth =0.1 
nbins=20
bins=np.arange(minVal, maxVal + binwidth, binwidth)
ratio_corrected = v_pt_corrected/v_pt_target
ratio_predicted = v_pt_predicted/v_pt_target

label_corrected = 'Jona recipy (mean={:5.3f}, std={:5.3f})'.format(np.mean(ratio_corrected), np.std(ratio_corrected))
label_predicted = 'Predicted by NN (mean={:5.3f}, std={:5.3f})'.format(np.mean(ratio_predicted), np.std(ratio_predicted))

plt.hist(ratio_predicted, bins=bins, color='tab:orange', label=label_predicted, histtype = 'step', density=True)
plt.hist(ratio_corrected, bins=bins, color='tab:cyan', label=label_corrected, histtype = 'step', density=True)
plt.legend(loc='upper left', fontsize='x-small')
plt.xlabel("")
plt.savefig(plotsDir+'hPtComparison.png')
plt.clf()

v_pz_measured = v_allData[:,35]
v_pz_corrected = v_allData[:,8]
v_pz_target = v_target[:,1]
v_pz_predicted = prediction_pz

ratio_pz_corrected = v_pz_corrected/v_pz_target
ratio_pz_predicted = v_pz_predicted/v_pz_target
label_corrected = 'Jona recipy (mean={:5.3f}, std={:5.3f})'.format(np.mean(ratio_pz_corrected), np.std(ratio_pz_corrected))
label_predicted = 'Predicted by NN (mean={:5.3f}, std={:5.3f})'.format(np.mean(ratio_pz_predicted), np.std(ratio_pz_predicted))

plt.hist(ratio_pz_predicted, bins=bins, color='tab:orange', label=label_predicted, histtype = 'step', density=True)
plt.hist(ratio_pz_corrected, bins=bins, color='tab:cyan', label=label_corrected, histtype = 'step', density=True)
plt.legend(loc='upper left', fontsize='x-small')
plt.xlabel("")
plt.savefig(plotsDir+'hPzComparison.png')
plt.clf()