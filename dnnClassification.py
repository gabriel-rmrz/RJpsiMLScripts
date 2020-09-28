#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import root_pandas

from utils.CMSStyle import *
from ROOT import gPad, TCanvas
from ROOT import TGraph

from root_pandas import read_root
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv1D, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN


def printReport():    
  pass

def plotROCCurve(test_target, test_predicted):
  '''
  Plot the ROC curve.
  fpr: false positive rates.
  tpr: true positive rates.
  '''
  c1 = TCanvas('c1', 'c1', 800, 600)
  CMS_lumi(gPad, 5, 1)
  from sklearn.metrics import roc_curve
  print(sum(test_target))

  fpr_signal, tpr_signal, threshold_signal = roc_curve(test_target, test_predicted, pos_label=None)
  roc_graph = TGraph(len(fpr_signal), fpr_signal, tpr_signal)
  roc_graph.SetTitle('ROC')
  roc_graph.GetXaxis().SetTitle('False positive rate')
  roc_graph.GetYaxis().SetTitle('True positive rate')
  roc_graph.Draw()
  c1.SaveAs("rocTest.pdf")
  return 0

inputFile = 'results/dnnFeedForward/mix_channel/results-mix_channel-nodes_100_100_100_100-vfrac_0p3-dropoutrate_0p1-bsize_200.root'
input_df = read_root(inputFile, 'tree')
#nn_inputFeatures = shuffle(nn_inputFeatures_tmp, random_state=0)
#  nn_q2_predicted            "nn_q2_predicted/F"            43920
#  nn_MuEnergyBcRestFrame     "nn_MuEnergyBcRestFrame/F"     43934
#  nn_missMass2               "nn_missMass2/F"               43914
#  nn_q2                      "nn_q2/F"                      43900
#  nn_missPt                  "nn_missPt/F"                  43908
#  nn_MuEnergyJpsiRestFrame   "nn_MuEnergyJpsiRestFrame/F"   43938
#  nn_varPt                   "nn_varPt/F"                   43906
#  nn_deltaRMu1Mu2            "nn_deltaRMu1Mu2/F"            43920
#  nn_MuPhi                   "nn_MuPhi/F"                   43906
#  nn_MuPt                    "nn_MuPt/F"                    43904
#  nn_MuEta                   "nn_MuEta/F"                   43906
inputFeaturesNames = ['nn_MuEnergyBcRestFrame', 'nn_MuEnergyJpsiRestFrame', 'nn_q2', 'nn_missMass2', 'nn_missPt', 'nn_varPt', 'nn_deltaRMu1Mu2', 'nn_MuPhi', 'nn_MuPt', 'nn_MuEta']
nn_inputFeatures = input_df[inputFeaturesNames].to_numpy()
#nn_inputFeatures = np.swapaxes(nn_inputFeatures, 1, 0)
nn_target = input_df['signalDecayPresent'].to_numpy()
print(nn_target.mean())
train_input, test_input, train_target, test_target = train_test_split(nn_inputFeatures, nn_target,
                                    #test_size=validation_frac,
                                    test_size=0.3,
                                    #train_size=5000,
                                    #test_size= 2000,
                                    random_state=9,
                                    shuffle=True,
                                    stratify=nn_target)  

print(test_input.shape)
print(train_input.shape)
print(test_target.shape)
print(train_target.shape)



batch_size = 30
n_epochs = 100
dropoutRate = 0.2

inputLayer = Input(shape=(len(inputFeaturesNames),))
x = BatchNormalization()(inputLayer)
x = Dense(10, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
x = Dense(8, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
x = Dense(8, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
x = Dense(10, activation = 'relu')(x)
x = Dropout(rate=dropoutRate)(x)
outputLayer = Dense(1, activation = 'sigmoid')(x)


model = Model(inputs = inputLayer, outputs = outputLayer)
model.compile(loss='binary_crossentropy', optimizer ='adam', metrics = ['accuracy'])
model.summary()

history = model.fit(train_input, 
        train_target,
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
#plt.show()

test_predicted = model.predict(test_input).ravel()

plotROCCurve(test_target=test_target, test_predicted=test_predicted)

test_predicted_int = [int( x > 0.5) for x in test_predicted]
print(sum(test_predicted_int)/len(test_predicted_int))
