import os
from featuresList import featuresListResults, featuresListInputs, featuresList

import numpy as np
import pandas as pd
from root_numpy import array2tree, array2root, fill_hist
from ROOT import TH2, TH2F, gROOT, TCanvas
import ROOT

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from pylab import savefig
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import randint
import copy
seed = 7
np.random.seed(seed)

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasRegressor
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
#trainingSample = 'tau'
trainingSample = 'muon'
#trainingSample = 'mix'

plotsDir = 'plots/dnnFeedForward/'+ trainingSample + '_channel/'

predictionSample = trainingSample
#predictionSample = 'tau'
#predictionSample = 'muon'
#predictionSample = 'mix'

inputFile="featuresData.npz"
f = np.load(inputFile)#, allow_pickle=True)
inputData = f["arr_0"]
inputData = np.swapaxes(inputData, 0, 1)
validation_frac = 0.3

inputData_tau = inputData[inputData[:,175].astype(int) == 1,] # 28 for signal channel and 29 for normalization channel
inputData_muon = inputData[inputData[:,176].astype(int) == 1,] # 28 for signal channel and 29 for normalization channel
inputData_norm = inputData[inputData[:,176].astype(int) == 1,] # 28 for signal channel and 29 for normalization channel
inputData_norm = inputData_norm[-3000:,:] 
inputData_sig = inputData[inputData[:,175].astype(int) == 1,] # 28 for signal channel and 29 for normalization channel
inputData_sig = inputData_sig[:1000,:] 
inputData_mix = np.append(inputData_norm, inputData_sig, axis=0)


inputData = inputData_tau
if(trainingSample == 'muon'):
  inputData = inputData_muon
elif(trainingSample == 'mix'):
  inputData = inputData_mix

predictionData = inputData_tau
if(predictionSample == 'muon'):
  predictionData = inputData_muon
elif(predictionSample == 'mix'):
  predictionData = inputData_mix

print(inputData.shape)
#target=pd.DataFrame(inputData[:,0]/inputData[:,102], columns=['bcPt_gen_reco_ratio'])

target=pd.DataFrame(np.reshape(inputData[:,0],[inputData[:,0].size, 1]), columns=['gen_bc_pt'])
sc_target = StandardScaler().fit(target)
inputFeatures = [ 
  71, 72, 73, 74, 75, 
  76, 77, 78, 79, 80, 81, 82, 83, 84,
  86, 87, 88, 89, 90, 91, 92, 93, 94,
  101, 102, 103, 104, 105,
  111, 112, 113, 114,
  121, 122, 123, 124, 125,
  131, 132, 133, 134,
  138, 139, 140, 141]
reg_input=inputData[:,inputFeatures]

#inputDataToDF = np.array([tuple(inputData[i,:]) for i in range(inputData[:,0].size)], dtype=featuresListInputs)
#df = pd.DataFrame(inputDataToDF)
df = pd.DataFrame(inputData, columns=featuresListInputs, dtype=np.float32)

sc = StandardScaler().fit(reg_input)

#t_allData, v_allData, t_target, v_target = train_test_split(inputData, target, 
t_allData, v_allData, t_target, v_target = train_test_split(df, target, 
                                    test_size=validation_frac, 
                                    random_state=8, 
                                    shuffle=True, 
                                    stratify=inputData[:,176]) # Stratify keep the proportion of signal and normalization samples in the splitted samples.
print(t_allData.shape)
print(v_allData.shape)
print(t_target.shape)
print(v_target.shape)
featuresListInputs = np.array(featuresListInputs)
t_data=t_allData[featuresListInputs[inputFeatures]]
v_data=v_allData[featuresListInputs[inputFeatures]]


scaled_t_data=sc.transform(t_data)
scaled_v_data=sc.transform(v_data)

scaled_t_target = sc_target.transform(t_target)
scaled_v_target = sc_target.transform(v_target)

batch_size = 50 
n_epochs = 3
dropoutRate = 0.1
inputData.shape

data_len=scaled_t_data.shape[1]

########################################################################
########################################################################
# Model definition
########################################################################
########################################################################
def build_regression_model(nInnerLayers, nNodes, dropoutRate):
  model = Sequential()
  model.add(Dense(45, activation="relu", kernel_initializer="glorot_uniform", input_dim=data_len))
  #model.add(Dropout(dropoutRate))
  for iLayer in range(nInnerLayers):
    if(nNodes[iLayer] != 0):
      model.add(Dense(nNodes[iLayer], activation="relu", kernel_initializer="glorot_uniform"))
      model.add(Dropout(dropoutRate))
  #model.add(Dense(100, activation="relu", kernel_initializer="glorot_uniform"))
  #model.add(Dropout(dropoutRate))
  #model.add(Dense(60, activation="relu", kernel_initializer="glorot_uniform"))
  #model.add(Dropout(dropoutRate))
  #model.add(Dense(40, activation="relu", kernel_initializer="glorot_uniform"))
  #model.add(Dropout(dropoutRate))
  #model.add(Dense(20, activation="relu", kernel_initializer="glorot_uniform"))
  #model.add(Dropout(dropoutRate))
  model.add(Dense(1, kernel_initializer="glorot_uniform"))
  print('compiling')
  model.compile(loss='mse', optimizer='adam',metrics=['mse', 'mae'])
  #model.summary()
  return model

#innerLayersNodes = [100, 60, 40, 20] 

histolist = []
nNodes= [20,40, 60, 80, 100]
#nNodes= [0]

for nInnerLayers in [1]:
  K.clear_session()
  for iNodes1 in nNodes:
#    for nNodes2 in [50]:#range(10,100,30):
    model = build_regression_model(nInnerLayers=nInnerLayers, nNodes=nNodes, dropoutRate=dropoutRate)
    history=model.fit(scaled_t_data , 
                      scaled_t_target, 
                      nb_epoch=n_epochs, verbose=1,batch_size=batch_size, 
                      validation_data=(scaled_v_data , scaled_v_target))
    histolist.append(pd.DataFrame(history.history, index=history.epoch))

historydf= pd.concat(histolist, axis=1)
metrics_reported = histolist[0].columns
idx = pd.MultiIndex.from_product([nNodes, metrics_reported],
                             names=['nNodes', 'metric'])
historydf.columns = idx
def plotHistory(historydf):
  plt.figure(num=None, figsize=(5, 5), dpi = 300, facecolor='w', edgecolor='k') 
  ax = plt.subplot(211)
  historydf.xs('loss', axis = 1, level = 'metric').plot(ax=ax)
  plt.title("Loss")
  ax = plt.subplot(212)
  historydf.xs('mean_squared_error', axis = 1, level = 'metric').plot(ax=ax)
  plt.title("MSE")
  plt.xlabel("Epochs")
  plt.tight_layout()
  plt.savefig(plotsDir+'loss_and_mse_history.png')
  plt.clf()

plotHistory(historydf)

plt.figure(num=None, figsize=(6, 5), dpi = 300, facecolor='w', edgecolor='k') 
minVal = 10. 
maxVal = 50.
binwidth =1 
bins=np.arange(minVal, maxVal + binwidth, binwidth)
v_target['gen_bc_pt'].plot.hist(bins=bins, histtype = 'step')
plt.title("gen_bc_pt")
plt.savefig(plotsDir+'targetDist.png')
plt.clf()

minVal = 10. 
maxVal = 50.
binwidth = 1 
bins=np.arange(minVal, maxVal + binwidth, binwidth)
v_allData['gen_b_pt'].plot.hist(bins=bins, histtype = 'step')
plt.title("genBc_pt")
plt.savefig(plotsDir+'oldTargetDist.png')
plt.clf()

plt.ylim(0., 2)

minVal = 10. 
maxVal = 50.
binwidth = 4 
bins=np.arange(minVal, maxVal + binwidth, binwidth)
v_allData_barrel = v_allData.query('abs(gen_jpsi_mu1_Eta)< 1.2 & abs(gen_jpsi_mu2_Eta) < 1.2')
v_allData_endcap = v_allData.query('abs(gen_jpsi_mu1_Eta)> 1.2 | abs(gen_jpsi_mu2_Eta) > 1.2')
bcPt_reco_gen_ratio_barrel = v_allData_barrel['Bc_pt']/v_allData_barrel['gen_b_pt']
bcPt_reco_gen_ratio_endcap = v_allData_endcap['Bc_pt']/v_allData_endcap['gen_b_pt']
bcPt_gen_barrel = v_allData_barrel['gen_b_pt']
bcPt_gen_endcap = v_allData_endcap['gen_b_pt']
#profilePlot = sns.regplot(x= v_allData['gen_b_pt'], y=v_target['bcPt_gen_reco_ratio'], x_bins=15, fit_reg=None)
profilePlot = sns.regplot(x= bcPt_gen_barrel, y=bcPt_reco_gen_ratio_barrel, label='barrel', color = 'black', x_bins=bins, fit_reg=None)
profilePlot = sns.regplot(x= bcPt_gen_endcap, y=bcPt_reco_gen_ratio_endcap, label='endcap', color = 'red', x_bins=bins, fit_reg=None)
profileFig = profilePlot.get_figure()
#profileFig.legend('lower center', fontsize='x-small')
plt.ylabel("recoBc_pt/ genBc_pt")
profileFig.legend()
profileFig.savefig(plotsDir+'profile.png')
plt.clf()


prediction_scaled_pt = model.predict(scaled_v_data)
prediction_pt = sc_target.inverse_transform(prediction_scaled_pt)
print(type(prediction_pt))
print(prediction_pt.shape)
#v_allData['bc_predictedPt'] = prediction_pt[:,0]
prediction_df = pd.DataFrame(prediction_pt, columns=['bc_pt'])
x = np.linspace(0., 60., 1000)
plt.figure(num=None, figsize=(6, 6), dpi = 300, facecolor='w', edgecolor='k') 

plt.scatter(v_target['gen_bc_pt'], prediction_df['bc_pt'], c='tab:orange', alpha = 0.2, label = 'NN prediction') 
plt.scatter(v_target['gen_bc_pt'], v_allData['bcCorrected_pt'], c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
#plt.scatter(v_target[:,0], prediction_pt, c='tab:red', alpha = 0.2) 
#plt.scatter(v_target[:,0], prediction_pt, c='tab:red', alpha = 0.2, label = 'corrected_data') 
plt.plot(x, x + 0, '-b', label= 'Truth MC')
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True) 
plt.xlabel("pt(Truth MC) [GeV]")
plt.ylabel("pt [GeV]")
plt.savefig(plotsDir+'sPtComparison.png')
plt.clf()

exit()
historydf[['loss', 'val_loss']].plot()
plt.title('loss')


plt.figure(num=None, figsize=(3, 3), dpi = 300, facecolor='w', edgecolor='k') 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(plotsDir+'loss_histo.png')
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


exit()

#x = np.linspace(-150., 150., 1000)
#plt.scatter(v_target[:,1], prediction_pz, c='tab:orange', alpha = 0.2, label = 'NN prediction') 
#plt.scatter(v_target[:,1], v_allData[:,8], c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
#plt.scatter(v_target[:,1], prediction_pz, c='tab:red', alpha = 0.2)
#plt.plot(x, x + 0, '-b', label= 'Truth MC')
#plt.legend(loc='upper left', fontsize='x-small')
#plt.grid(True)
#plt.xlabel("pz(Truth MC) [GeV]")
#plt.ylabel("pz(NN prediction) [GeV]")
#plt.savefig(plotsDir+'sPzComparison.png')
#plt.clf()

x = np.linspace(0., 200., 1000)
#plt.scatter(v_target[:,2], prediction_E, c='tab:orange', alpha = 0.2, label = 'NN prediction') 
#plt.scatter(v_target[:,2], v_allData[:,9], c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
plt.plot(x, x + 0, '-b', label= 'Truth MC')
#plt.scatter(v_target[:,2], prediction_E, c='tab:blue', alpha = 0.2)
#plt.scatter(v_target[:,2], prediction_E, c='tab:red', alpha = 0.2)
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True)
plt.xlabel("Energy(Truth MC) [GeV]")
plt.ylabel("Energy(NN prediction) [GeV]")
#plt.savefig(plotsDir+'sEnergyComparison.png')
plt.clf()

minVal = 0
maxVal = 60
binwidth = 4
bins=range(minVal, maxVal + binwidth, binwidth)
#plt.hist(v_data[:,4], bins=bins, color='tab:gray', label='Non corrected', histtype = 'step', density=True)
#plt.hist(prediction_E, bins=bins, color='tab:orange', label='NN prediction', histtype = 'step', density=True)
#plt.hist(v_target[:,2], bins=bins,  color='tab:cyan', label='Jona recipy', histtype = 'step', density=True)
plt.legend(loc='upper right', fontsize='x-small')
#plt.xlim(-50,50)
#plt.grid(True)
plt.xlabel("Energy [GeV]")
#plt.savefig(plotsDir+'hEnergyComparison.png')
plt.clf()




########################################################################3
v_pt_measured = v_allData['Bc_pt']
v_pt_corrected = v_allData['bcCorrected_pt']
v_pt_target = v_target*v_allData['Bc_pt']
v_pt_predicted = prediction_pt['prediction_p4']*v_allData['Bc_pt']



minVal = 0. 
maxVal = 2.
binwidth =0.1 
nbins=20
bins=np.arange(minVal, maxVal + binwidth, binwidth)
ratio_corrected = v_pt_corrected/v_pt_target
ratio_predicted = v_pt_predicted/v_pt_target
print(ratio_predicted.shape)
label_corrected = 'Jona recipy (mean={:5.3f}, std={:5.3f})'.format(np.mean(ratio_corrected), np.std(ratio_corrected))
label_predicted = 'Predicted by NN (mean={:5.3f}, std={:5.3f})'.format(np.mean(ratio_predicted), np.std(ratio_predicted))
plt.hist(ratio_predicted, bins=bins, color='orange', label=label_predicted, histtype = 'step', density=True)
plt.hist(ratio_corrected, bins=bins, color='tab:cyan', label=label_corrected, histtype = 'step', density=True)
plt.legend(loc='upper left', fontsize='x-small')
plt.savefig(plotsDir+'hPtComparison.png')
plt.clf()

#v_pz_measured = v_allData[:,35]
#v_pz_corrected = v_allData[:,8]
#v_pz_target = v_target[:,1]
#v_pz_predicted = prediction_pz

#ratio_pz_corrected = v_pz_corrected/v_pz_target
#ratio_pz_predicted = v_pz_predicted/v_pz_target
#label_corrected = 'Jona recipy (mean={:5.3f}, std={:5.3f})'.format(np.mean(ratio_pz_corrected), np.std(ratio_pz_corrected))
#label_predicted = 'Predicted by NN (mean={:5.3f}, std={:5.3f})'.format(np.mean(ratio_pz_predicted), np.std(ratio_pz_predicted))

#plt.hist(ratio_pz_predicted, bins=bins, color='tab:orange', label=label_predicted, histtype = 'step', density=True)
#plt.hist(ratio_pz_corrected, bins=bins, color='tab:cyan', label=label_corrected, histtype = 'step', density=True)
#plt.legend(loc='upper left', fontsize='x-small')
#plt.xlabel("")
#plt.savefig(plotsDir+'hPzComparison.png')
#plt.clf()




plt.scatter(v_target*v_allData['Bc_pt'], ratio_predicted, c='tab:orange', alpha = 0.2, label = 'NN prediction') 
plt.scatter(v_target*v_allData['Bc_pt'], ratio_corrected, c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
plt.legend(loc='upper left', fontsize='x-small')
plt.grid(True)
plt.ylabel("Ratio pT(Bc) (modified/Truth MC) ")
plt.xlabel("pT(Bc) (Truth MC) [GeV]")
plt.savefig(plotsDir+'sRatioVsPt.png')
plt.clf()

#plt.scatter(v_target[:,1], ratio_pz_predicted, c='tab:orange', alpha = 0.2, label = 'NN prediction') 
#plt.scatter(v_target[:,1], ratio_pz_corrected, c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
#plt.legend(loc='upper left', fontsize='x-small')
#plt.grid(True)
##plt.xlim(-10,10)
#plt.ylim(0.,2.)
#plt.ylabel("Ratio pT(Bc) (modified/Truth MC) ")
#plt.xlabel("pT(Bc) (Truth MC) [GeV]")
#plt.savefig(plotsDir+'sRatioVsPz.png')
#plt.clf()

#plt.scatter(v_target[ratio_pz_predicted<0.,1], ratio_pz_predicted[ratio_pz_predicted < 0.], c='tab:orange', alpha = 0.2, label = 'NN prediction') 
#plt.scatter(v_target[ratio_pz_corrected<0.,1], ratio_pz_corrected[ratio_pz_corrected < 0.], c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
#plt.scatter(v_target[ratio_pz_predicted>2.,1], ratio_pz_predicted[ratio_pz_predicted >2.], c='tab:orange', alpha = 0.2) 
#plt.scatter(v_target[ratio_pz_corrected>2.,1], ratio_pz_corrected[ratio_pz_corrected >2.], c='tab:cyan', alpha = 0.2) 
#print(ratio_pz_predicted[ratio_pz_predicted<0.])
#plt.legend(loc='upper left', fontsize='x-small')
#plt.xlim(-10,10)
#plt.ylim(-200,200)
#plt.grid(True)
#plt.ylabel("Ratio pz(Bc) (modified/Truth MC) ")
#plt.xlabel("pz(Bc) (Truth MC) [GeV]")
#plt.savefig(plotsDir+'sRatioVsPz_extremeVals.png')
#plt.clf()

plt.scatter(v_target[ratio_predicted<0.], ratio_predicted[ratio_predicted < 0.], c='tab:orange', alpha = 0.2, label = 'NN prediction') 
plt.scatter(v_target[ratio_corrected<0.], ratio_corrected[ratio_corrected < 0.], c='tab:cyan', alpha = 0.2, label = 'Jona recipy') 
plt.scatter(v_target[ratio_predicted>2.], ratio_predicted[ratio_predicted >2.], c='tab:orange', alpha = 0.2) 
plt.scatter(v_target[ratio_corrected>2.], ratio_corrected[ratio_corrected >2.], c='tab:cyan', alpha = 0.2) 
plt.legend(loc='upper left', fontsize='x-small')
plt.xlim(-10,10)
plt.ylim(-200,200)
plt.grid(True)
plt.ylabel("Ratio pt(Bc) (modified/Truth MC) ")
plt.xlabel("pt(Bc) (Truth MC) [GeV]")
plt.savefig(plotsDir+'sRatioVsPt_extremeVals.png')
plt.clf()

exit()

r_allData = np.append(prediction_pt.reshape(prediction_pt.size,1), v_allData, axis=1)
r_allData = np.append(ratio_predicted.reshape(ratio_predicted.size,1), r_allData, axis=1)
r_allData = np.append(ratio_corrected.reshape(ratio_corrected.size,1), r_allData, axis=1)
allDataToTree = np.array([tuple(r_allData[i,:]) for i in range(r_allData[:,0].size)], dtype=featuresListResults)
dataTree = array2root(allDataToTree, 'test.root', mode='recreate')

hCorrected = TH2F('hCorrected', '', 30, 0., 60., 20, 0., 2.)
hPredicted = TH2F('hPredicted', '', 30, 0., 60., 20, 0., 2.)
#print(r_allData[:,[3,0]])
fill_hist(hCorrected, r_allData[:,[3,0]])
fill_hist(hPredicted, r_allData[:,[3,1]])

profileCorrected = hCorrected.ProfileX('Jona Correction')
profilePredicted = hPredicted.ProfileX('NN prediction')

c1 = TCanvas('c1', 'c1', 700, 500)
profileCorrected.SetLineColor(ROOT.kBlue)
profilePredicted.SetLineColor(ROOT.kOrange)
profileCorrected.GetXaxis().SetTitle('pT [GeV]')
profileCorrected.GetYaxis().SetRangeUser(0., 2.)
profileCorrected.GetYaxis().SetTitle('Ratio pT corrected/pt Truth MC' )

profileCorrected.Draw()
ROOT.gStyle.SetStatX(0.3)
ROOT.gStyle.SetOptStat(0)
profilePredicted.Draw("SAME")
c1.SaveAs('test.png')
