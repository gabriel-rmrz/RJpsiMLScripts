import os
import argparse


import numpy as np
import pandas as pd
import root_pandas
from root_numpy import array2tree, array2root, fill_hist
from ROOT import TF1, TH2, TH2F, gROOT, TCanvas
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

def get_parameters():
  
  CLI = argparse.ArgumentParser()
  CLI.add_argument(
          "--nodes",
          nargs="*",
          type=int,
          default = [50, 50],
          )
  CLI.add_argument(
          "--valFrac",
          nargs="*",
          type=float,
          default = [0.3],
          )
  CLI.add_argument(
          "--testFrac",
          nargs="*",
          type=float,
          default = [0.3],
          )
  CLI.add_argument(
          "--batchSize",
          nargs="*",
          type=int,
          default = [200],
          )
  CLI.add_argument(
          "--nMaxEpochs",
          nargs="*",
          type=int,
          default =200,
          )
  CLI.add_argument(
          "--dropoutRate",
          nargs="*",
          type=float,
          default = [0.1],
          )
  CLI.add_argument(
          "--trainingSample",
          nargs="?",
          type=str,
          default = 'mix',
          )
  args = CLI.parse_args()

  print(type(args.valFrac))
  print(args.testFrac)
  print(args.batchSize)
  print(args.dropoutRate)



  validation_frac = (args.valFrac)[0]
  test_frac = (args.testFrac)[0]
  batch_size = (args.batchSize)[0]
  n_epochs = args.nMaxEpochs
  dropoutRate = (args.dropoutRate)[0]
  nodes= np.array(args.nodes) 
  trainingSample = args.trainingSample # Possible values: 'tau', 'muon', 'mix'

  #print("This is the output: %r" % args.nodes)
  print("This is the output: %s" % args.trainingSample)
  print("This is the output: %2.1f" % validation_frac)
  print("This is the output: %2.1f" % dropoutRate)
  print("This is the output: %d" % batch_size)
  print("This is the output: %d" % n_epochs)
  print("-vfrac_0p%d-dropoutrate_0p%d-bsize_%d" % ((int)(10*validation_frac), (int)(10*dropoutRate), batch_size))

  return validation_frac, test_frac, batch_size, n_epochs, dropoutRate, nodes, trainingSample

def get_profile_pdf(bc_pt):
  hPtRecoVsPtGen = TH2F('hPtRecoVsPtGen', '', 80, 0., 80., 80, 0., 80.)
  fill_hist(hPtRecoVsPtGen, bc_pt)
  profilePtRecoVsPtGen = hPtRecoVsPtGen.ProfileX()
  p1 = TF1("p1", "pol1", 10, 70)
  profilePtRecoVsPtGen.Fit("p1")
 
  print(p1(10))
  #print(np.array(list(map(p1,bc_pt[:,0]))))

  c1 = TCanvas('c1', 'c1', 700, 500)
  profilePtRecoVsPtGen.Draw()
  c1.SaveAs('profile.png')
  return p1

def get_target_distribution():
  pass

def get_ptDistribution_fromPrediction():
  pass

def get_data_frames(trainingSample, predictionSample,  allFeaturesList, inputFeatures):
  inputFile="data/featuresData.npz"
  f = np.load(inputFile)
  inputData = f["arr_0"]
  inputData = np.swapaxes(inputData, 0, 1)
  
  #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  #    print(inputData[:,175])
  #exit()
  ##175 for signal channel and 176 for normalization channel
  inputData_tau = inputData[inputData[:,175].astype(int) == 1,]
  inputData_muon = inputData[inputData[:,176].astype(int) == 1,]
  inputData_norm = inputData[inputData[:,176].astype(int) == 1,]
  nSignalInMix = (int)(inputData_muon[:,0].size /3)
  inputData_sig = inputData[inputData[:,175].astype(int) == 1,]
  inputData_sig = inputData_sig[:nSignalInMix,:] 
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
  
  allFeatures = inputData
  profilePdf =  get_profile_pdf(inputData[:, [102,0]])
  
  #targetDenominator = np.array(list(map(profilePdf,inputData[:,102])))
  #targetNominator = np.array(inputData[:,0])
  #targetFeatures = targetNominator.reshape(-1,1)/ targetDenominator.reshape(-1,1)

  #targetFeatures = np.reshape((inputData[:,0] +5)/inputData[:,102],[inputData[:,0].size, 1])
  #targetFeatures = np.reshape((inputData[:,0])/inputData[:,102],[inputData[:,0].size, 1])
  targetFeatures = np.reshape(inputData[:,0],[inputData[:,0].size, 1])
  #mu, sigma = 0., 1.
  #targetFeatures = np.reshape(np.random.normal(mu, sigma, inputData[:,0].size), inputData[:,0].size, 1)

  inputFeatures = inputData[:,inputFeatures]

  #return allFeatures[targetFeatures[:,0] > 1.2,:], inputFeatures[targetFeatures[:,0] > 1.2,:], targetFeatures[targetFeatures[:,0] > 1.2]
  return allFeatures, inputFeatures, targetFeatures
  
########################################################################
########################################################################
# Model definition
########################################################################
########################################################################

def build_regression_model(nodes, dropoutRate, data_len):
  model = Sequential()
  #model.add(Dense(nodes[0], activation="relu", kernel_initializer="glorot_uniform", input_dim=data_len))
  model.add(Dense(nodes[0], activation="relu", kernel_initializer="lecun_normal", input_dim=data_len))
  model.add(Dropout(dropoutRate))
  for iNode in nodes[1:]:
    #model.add(Dense(iNode, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(iNode, activation="relu", kernel_initializer="lecun_normal"))
    model.add(Dropout(dropoutRate))
  #model.add(Dense(1, kernel_initializer="glorot_uniform"))
  model.add(Dense(1, kernel_initializer="lecun_normal"))
  print('compiling')
  model.compile(loss='mse', optimizer='adam',metrics=['mse', 'mae'])
  model.summary() 
  return model

def main():
  from utils.featuresList import outputFeaturesList, allFeaturesList, featuresList
  validation_frac, test_frac, batch_size, n_epochs, dropoutRate, nodes, trainingSample = get_parameters()

  ########################################################################
  ########################################################################
  # Preparation of the input featues
  ########################################################################
  ########################################################################
  
  plotsDir = 'plots/dnnFeedForward/'+ trainingSample + '_channel/'
  resultsDir = 'results/dnnFeedForward/' + trainingSample + '_channel/'
  
  predictionSample = trainingSample

  allFeaturesList = np.array(allFeaturesList)
  inputFeatures = [ 
    71, 72, 73, 74, 75, 
    76, 77, 78, 79, 80, 81, 82, 83, 84,
    86, 87, 88, 89, 90, 91, 92, 93, 94,
    101, 102, 103, 104, 105,
    111, 112, 113, 114,
    121, 122, 123, 124, 125,
    131, 132, 133, 134,
    138, 139, 140, 141]
  allData, inputData, targetData = get_data_frames(trainingSample=trainingSample, predictionSample=predictionSample, allFeaturesList=allFeaturesList, inputFeatures=inputFeatures)

  train_allData, test_allData, train_target, test_target = train_test_split(allData, targetData, 
                                      #test_size=validation_frac) 
                                      #test_size=test_frac, 
                                      train_size=5000,
                                      test_size= 2000,
                                      random_state=9, 
                                      shuffle=True,
                                      stratify=allData[:,176])
  
  t_allData, v_allData, t_target, v_target = train_test_split(train_allData, train_target, 
                                      #test_size=validation_frac, 
                                      train_size=3000,
                                      test_size=2000, #validation_frac, 
                                      random_state=8, 
                                      shuffle=True,
                                      stratify=train_allData[:,176])
  t_input=t_allData[:, inputFeatures]
  v_input=v_allData[:, inputFeatures]
  test_input = test_allData[:, inputFeatures]

  sc = StandardScaler().fit(inputData)
  scaled_t_input=sc.transform(t_input)
  scaled_v_input=sc.transform(v_input)
  scaled_test_input=sc.transform(test_input)

  sc_target = StandardScaler().fit(targetData)
  scaled_t_target=sc_target.transform(t_target)
  scaled_v_target=sc_target.transform(v_target)
  scaled_test_target=sc_target.transform(test_target)

  #######
  data_len=scaled_t_input.shape[1]
  K.clear_session()
  model = build_regression_model(nodes=nodes, dropoutRate=dropoutRate, data_len=data_len)
  history=model.fit(scaled_t_input , 
                    scaled_t_target, 
                    #t_target, 
                    epochs=n_epochs, verbose=1, batch_size=batch_size, 
                    #validation_data=(scaled_v_input , v_target),
                    validation_data=(scaled_v_input , scaled_v_target),
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience = 10, verbose=1),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1),
                        TerminateOnNaN()]
                    )
  
  history_df = pd.DataFrame(history.history, index=history.epoch)

  profilePdf =  get_profile_pdf(test_allData[:, [102,0]])
  targetDenominator = np.array(list(map(profilePdf,test_allData[:,102])))
  targetDenominator = np.reshape(targetDenominator, [targetDenominator.size, 1])
  #prediction_ratio_pt = model.predict(scaled_test_input)
  #prediction_pt = prediction_ratio_pt *targetDenominator
  scaled_prediction_ratio_pt = model.predict(scaled_test_input)
  prediction_pt = sc_target.inverse_transform(scaled_prediction_ratio_pt)
  target_ratio_df = pd.DataFrame(test_target, columns=['target'])
  prediction_ratio_df = pd.DataFrame(prediction_pt, columns=['pt_predicted'])
  ratio_out_test_df = pd.concat([target_ratio_df, prediction_ratio_df], axis=1)
  ratio_out_test_df.to_root('tmp/ratios.root', key='tree')
  
  prediction_pt_df = pd.DataFrame(prediction_pt[:,0], columns=['bc_pt_predicted'])
  prediction_pt_ratio_df = pd.DataFrame(prediction_pt[:,0]/test_allData[:,0], columns=['bc_ptRatio_predictedGen'])
  #import scipy
  #print(scipy.stats.skew(prediction_ratio_pt[:,0]))
  
  corrected_pt_ratio_df = pd.DataFrame(test_allData[:,56]/(test_allData[:,0]), columns=['bc_ptRatio_correctedGen'])
  
  resultsDir = 'results/dnnFeedForward/' + trainingSample + '_channel/'
  outputFile = resultsDir + "results-"+trainingSample + "_channel-nodes"
  historyFile = resultsDir + "history-"+trainingSample + "_channel-nodes"
  for node in nodes:
      outputFile += "_%d" % node
      historyFile += "_%d" % node

  outputFile += "-vfrac_0p%d-dropoutrate_0p%d-bsize_%d" % ((int)(10*validation_frac), (int)(10*dropoutRate), batch_size)
  historyFile += "-vfrac_0p%d-dropoutrate_0p%d-bsize_%d" % ((int)(10*validation_frac), (int)(10*dropoutRate), batch_size)

  outputFile += ".root"
  historyFile += ".root"
  

  test_allDF = pd.DataFrame(test_allData, columns=allFeaturesList, dtype=np.float32)
  results_df = pd.concat([prediction_pt_df, prediction_pt_ratio_df, corrected_pt_ratio_df, test_allDF], axis=1)
  results_df.to_root(outputFile, key='tree')
  history_df.to_root(historyFile, key='historyTree')
  #predictionLabel = "nLay_%d-nNod_%d--Mean= %5.3f, RMS= %5.3f" % (Len(nodes), nodes, prediction_pt_ratio_df['ratio_bc_pt'].mean(), prediction_pt_ratio_df['ratio_bc_pt'].std()))

if __name__=="__main__":
    main()
