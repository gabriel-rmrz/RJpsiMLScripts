import os
import argparse

import math
import numpy as np
import pandas as pd
import root_pandas
import ROOT
from root_numpy import array2tree, array2root, fill_hist
from ROOT import TF1, TH2, TH2F, gROOT, TCanvas

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from pylab import savefig
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import inv_boxcox
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

def get_target_distributions_scaler(bc_pt_gen):
  bc_pt_gen = np.reshape(bc_pt_gen,[bc_pt_gen.size, 1])
  sc_target = StandardScaler().fit(bc_pt_gen)
  targetData=sc_target.transform(bc_pt_gen)
  return targetData

def get_pt_from_NNPrediction_scaler(prediction_nn, bc_pt_gen):
  bc_pt_gen = np.reshape(bc_pt_gen,[bc_pt_gen.size, 1])
  sc_target = StandardScaler().fit(bc_pt_gen)
  prediction_pt = sc_target.inverse_transform(prediction_nn[:,0])
  return prediction_pt

def get_target_distributions_ptRatio(bc_pt_gen, bc_pt_reco):
  targetData = np.reshape((bc_pt_gen + 5.)/bc_pt_reco,[bc_pt_gen.size, 1])
  return targetData

def get_pt_from_NNPrediction_ptRatio(prediction_nn, bc_pt_reco):
  prediction_pt = (prediction_nn[:,0] * bc_pt_reco) - 5
  return prediction_pt

def get_target_distributions_pxpypz_scaler(bc_p3_gen):
  #bc_p3_gen = np.reshape(bc_p3_gen,[bc_p3_gen[:,0].size, 1])
  sc_target = StandardScaler().fit(bc_p3_gen)
  targetData=sc_target.transform(bc_p3_gen)
  return targetData

def get_pxpypz_from_NNPrediction_pxpypz_scaler(prediction_nn, bc_p3_gen):
  bc_p3_gen = np.reshape(bc_p3_gen,[bc_p3_gen[:,0].size, 3])
  sc_target = StandardScaler().fit(bc_p3_gen)
  prediction_p3 = sc_target.inverse_transform(prediction_nn)
  return prediction_p3

def get_target_distributions_pt_scaler_etaphi(bc_p3_ptetaphi_gen):
  pt = np.reshape(bc_p3_ptetaphi_gen[:,0], [bc_p3_ptetaphi_gen[:,0].size,1])
  sc_target = StandardScaler().fit(pt)
  pt_scaled = sc_target.transform(pt)
  pt_scaled = np.reshape(pt_scaled,[pt_scaled.size, 1])
  targetData = np.append(pt_scaled, bc_p3_ptetaphi_gen[:,[1,2]], axis=1)
  return targetData

def get_pxpypz_from_NNPrediction_pt_scaler_etaphi(prediction_nn, bc_p3_ptetaphi_gen):
  print('Getting px, py and pz')
  bc_pt_gen = np.reshape(bc_p3_ptetaphi_gen[:,0],[bc_p3_ptetaphi_gen[:,0].size, 1])
  sc_target = StandardScaler().fit(bc_pt_gen)
  pt_prediction = sc_target.inverse_transform(prediction_nn[:,0])
  #pt_prediction = np.reshape(pt_prediction, [pt_prediction.size, 1])

  px = pt_prediction * np.cos(prediction_nn[:,2])
  py = pt_prediction * np.sin(prediction_nn[:,2])
  pz = pt_prediction * np.sinh(prediction_nn[:,1])

  print(prediction_nn[:,1])

  px = np.reshape(px, [px.size,1])
  py = np.reshape(py, [py.size,1])
  pz = np.reshape(pz, [pz.size,1])
  
  prediction_p2 = np.append(px, py, axis=1)
  prediction_p3 = np.append(prediction_p2, pz, axis=1)
  #print(prediction_p3.shape)
  #exit()
  
  print('Done getting px, py and pz')
  return prediction_p3
  
def get_target_distributions_boxcox(t_target, v_target, test_target,targetData):
  #xp, expp = stats.boxcox(allData[:,0])
  #targetData_pre = xp/allData[:,102]
  #targetData = np.reshape(targetData_pre,[targetData_pre.size, 1])
  pass
def get_pt_from_NNPrediction_boxcox(prediction_nn, targetData):
  #prediction_ptRatio = inv_boxcox(prediction_nn[:,0], expp)
  #prediction_pt  = prediction_ptRatio*test_allData[:,102]
  pass
  
def get_arrays(trainingSample, predictionSample,  allFeaturesList, inputFeatures):
  inputFile="data/featuresData.npz"
  f = np.load(inputFile)
  inputData = f["arr_0"]
  inputData = np.swapaxes(inputData, 0, 1)
  
  #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  #    print(inputData[:,175])
  ##175 for signal channel and 176 for normalization channel
  inputData_tau = inputData[inputData[:,175].astype(int) == 1,]
  inputData_muon = inputData[inputData[:,176].astype(int) == 1,]
  inputData_norm = inputData[inputData[:,176].astype(int) == 1,]
  nSignalInMix = (int)(inputData_muon[:,0].size /3)
  inputData_sig = inputData[inputData[:,175].astype(int) == 1,]
  inputData_sig = inputData_sig[:nSignalInMix,:] 
  inputData_mix = np.append(inputData_norm, inputData_sig, axis=0)

  
  
  allData = inputData_tau
  if(trainingSample == 'muon'):
    allData = inputData_muon
  elif(trainingSample == 'mix'):
    allData = inputData_mix
  
  predictionData = inputData_tau
  if(predictionSample == 'muon'):
    predictionData = inputData_muon
  elif(predictionSample == 'mix'):
    predictionData = inputData_mix
  
  profilePdf =  get_profile_pdf(allData[:, [102,0]])
  
  #targetData = np.reshape(allData[:,0],[allData[:,0].size, 1])
  #targetData = get_target_distributions_ptRatio(allData[:,0],allData[:,102])
  #targetData = get_target_distributions_scaler(allData[:,0])
  #targetData = get_target_distributions_pxpypz_scaler(allData[:,[1,2,3]])
  #targetData = get_target_distributions_pt_scaler_etaphi(allData[:,[0, 5, 6]]) #taking bc_pt, bc_eta and bc_phi as inputs
  targetData = allData[:,63]

  diff_vertices = allData[:, [76, 77, 78]] - allData[:, [86, 87, 88]]
  #distance_Sec_Prim_vertices =  (diff_vertices[:,0]**2 + diff_vertices[:,1]**2 + diff_vertices[:,2]**2)** 0.5
  #distance_Sec_Prim_vertices = np.reshape(distance_Sec_Prim_vertices, [distance_Sec_Prim_vertices.size, 1])
  #allData = np.append(allData, distance_Sec_Prim_vertices, axis=1)
  allData = np.append(allData, diff_vertices, axis=1)

  inputData_nn = allData[:,inputFeatures]

  print(allData.shape)
  print(targetData.shape)

  train_allData, test_allData, train_target, test_target = train_test_split(allData, targetData, 
                                      #test_size=validation_frac,
                                      test_size=0.3, 
                                      #train_size=5000,
                                      #test_size= 2000,
                                      random_state=9, 
                                      shuffle=True,
                                      stratify=allData[:,176])
  
  t_allData, v_allData, t_target, v_target = train_test_split(train_allData, train_target, 
                                      test_size=0.3, 
                                      #train_size=3000,
                                      #test_size=2000, #validation_frac, 
                                      random_state=8, 
                                      shuffle=True,
                                      stratify=train_allData[:,176])
  t_input=t_allData[:, inputFeatures]
  v_input=v_allData[:, inputFeatures]
  test_input = test_allData[:, inputFeatures]

  sc = StandardScaler().fit(inputData_nn)
  scaled_t_input=sc.transform(t_input)
  scaled_v_input=sc.transform(v_input)
  scaled_test_input=sc.transform(test_input)

  return allData, scaled_t_input, t_target, scaled_v_input, v_target, scaled_test_input, test_target, test_allData
  
########################################################################
########################################################################
# Model definition
########################################################################
########################################################################

def build_regression_model(nodes, dropoutRate, data_len):
  model = Sequential()
  #kernel_ini = "glorot_uniform"
  kernel_ini = "lecun_normal"
  model.add(Dense(nodes[0], activation="relu", kernel_initializer=kernel_ini, input_dim=data_len))
  model.add(Dropout(dropoutRate))
  for iNode in nodes[1:]:
    model.add(Dense(iNode, activation="relu", kernel_initializer=kernel_ini))
    model.add(Dropout(dropoutRate))
  model.add(Dense(1, kernel_initializer=kernel_ini))
  print('compiling')
  model.compile(loss='mse', optimizer='adam',metrics=['mse', 'mae'])
  model.summary() 
  return model

def main():
  from utils.featuresList import allFeaturesList
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
    71, 72, 73, 74, 75, #bc, jpsi, mu1, mu2, unpairedmu energies
    76, 77, 78, 79, 80, 81, 82, 83, 84, #primary vertex + error
    86, 87, 88, 89, 90, 91, 92, 93, 94, #sv + error
    101, 102, 103, 104, 105, #bc mass, pt, px, py, pz
    111, 112, 113, 114, # unpaired mu pt, px, py, pz
    121, 122, 123, 124, 125, #bc mass, pt, px, py, pz
    131, 132, 133, 134, #mu1 pt, px, py, pz
    138, 139, 140, 141, #mu2 pt, px, py, pz
    -3, -2, -1] #vertex distance
  allData, t_input, t_target, v_input, v_target, test_input, test_target, test_allData = get_arrays(trainingSample=trainingSample, predictionSample=predictionSample, allFeaturesList=allFeaturesList, inputFeatures=inputFeatures)


  #######
  data_len=t_input.shape[1]
  K.clear_session()
  model = build_regression_model(nodes=nodes, dropoutRate=dropoutRate, data_len=data_len)
  history=model.fit(t_input , 
                    t_target, 
                    epochs=n_epochs, verbose=1, batch_size=batch_size, 
                    validation_data=(v_input , v_target),
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience = 10, verbose=1),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1),
                        TerminateOnNaN()]
                    )
  
  history_df = pd.DataFrame(history.history, index=history.epoch)

  profilePdf =  get_profile_pdf(test_allData[:, [102,0]])
  prediction_q2 = model.predict(test_input)
  #prediction_nn = model.predict(test_input)
  #prediction_pt = get_pt_from_NNPrediction_scaler(prediction_nn, allData[:, 0])
  #prediction_pt = get_pt_from_NNPrediction_ptRatio(prediction_nn, test_allData[:,102])
  #prediction_pxpypz=get_pxpypz_from_NNPrediction_pxpypz_scaler(prediction_nn, allData[:,[1,2,3]])
  #prediction_pxpypz=get_pxpypz_from_NNPrediction_pt_scaler_etaphi(prediction_nn, allData[:,[0,5,6]])

  #prediction_pt_df = pd.DataFrame(prediction_pt, columns=['bc_pt_predicted'])
  #prediction_pt_ratio_df = pd.DataFrame(prediction_pt/test_allData[:,0], columns=['pt_ptRatio_predicted'])
  #prediction_pxpypz_df = pd.DataFrame(prediction_pxpypz, columns=['bc_px_predicted', 'bc_py_predicted', 'bc_pz_predicted'])
  prediction_q2_df = pd.DataFrame(prediction_q2, columns=['nn_q2_predicted'])
  corrected_pt_ratio_df = pd.DataFrame(test_allData[:,56]/(test_allData[:,0]), columns=['bc_ptRatio_corrected'])
  
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
  #results_df = pd.concat([prediction_pt_df, prediction_pt_ratio_df, corrected_pt_ratio_df, test_allDF], axis=1)
  #results_df = pd.concat([prediction_pxpypz_df, corrected_pt_ratio_df, test_allDF], axis=1)
  results_df = pd.concat([prediction_q2_df, corrected_pt_ratio_df, test_allDF], axis=1)
  results_df.to_root(outputFile, key='tree')
  history_df.to_root(historyFile, key='historyTree')
  #predictionLabel = "nLay_%d-nNod_%d--Mean= %5.3f, RMS= %5.3f" % (Len(nodes), nodes, prediction_pt_ratio_df['ratio_bc_pt'].mean(), prediction_pt_ratio_df['ratio_bc_pt'].std()))

if __name__=="__main__":
    main()
