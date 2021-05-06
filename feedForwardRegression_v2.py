import os
import argparse

import math
import numpy as np
import pandas as pd
import root_pandas
import ROOT
from root_pandas import read_root
from root_numpy import array2tree, array2root, fill_hist
from ROOT import TF1, TH2, TH2F, gROOT, TCanvas

#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#import seaborn as sns
#from pylab import savefig
#from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from scipy import stats
#from scipy.special import inv_boxcox
from random import randint
import copy
seed = 7
np.random.seed(seed)

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout#, Flatten, Convolution2D, merge, Convolution1D
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

#%tensorflow_version 2.x
#%tensorflow_version 1.x
#print(tf.__version__)

def get_parameters():
  
  CLI = argparse.ArgumentParser()
  CLI.add_argument(
          "--nodes",
          nargs="*",
          type=int,
          default = [100, 100, 100, 100, 100],
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
          default = [100],
          )
  CLI.add_argument(
          "--nMaxEpochs",
          nargs="*",
          type=int,
          default =100,
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
  #xp, expp = stats.boxcox(allFeatures[:,0])
  #targetData_pre = xp/allFeatures[:,102]
  #targetData = np.reshape(targetData_pre,[targetData_pre.size, 1])
  pass
def get_pt_from_NNPrediction_boxcox(prediction_nn, targetData):
  #prediction_ptRatio = inv_boxcox(prediction_nn[:,0], expp)
  #prediction_pt  = prediction_ptRatio*test_allFeatures[:,102]
  pass
  
def get_arrays(trainingSample, predictionSample, inputFeatures):
  #inputFile = '/gpfs/ddn/srm/cms/store/user/garamire/ntuples/2021Jan29/BcToJpsiMuNu_UL_2021Jan26.root'
  #inputFile = "/gpfs/ddn/srm/cms/store/user/garamire/ntuples/2021Mar23/BcToJPsiMuMu_is_jpsi_mu_merged.root"
  inputFile = "data/BcToJPsiMuMu_is_jpsi_lepton_test.root"
  inputTree = 'BTo3Mu'
  allFeatures_df = read_root(inputFile, inputTree)
  target_df = allFeatures_df['mu1_grandmother_pt'].copy() #TODO: Change this to the k gen pt info once is included in the gen file

  train_allFeatures_df, test_allFeatures_df, train_target_df, test_target_df = train_test_split(allFeatures_df, target_df, 
                                      #test_size=validation_frac,
                                      test_size=0.3, 
                                      #train_size=50000,
                                      #test_size= 20000,
                                      random_state=9, 
                                      shuffle=True,
                                      stratify=allFeatures_df['is_signal_channel'])
  
  t_allFeatures_df, v_allFeatures_df, t_target_df, v_target_df = train_test_split(train_allFeatures_df, train_target_df, 
                                      test_size=0.3, 
                                      #train_size=30000,
                                      #test_size=20000, #validation_frac, 
                                      random_state=8, 
                                      shuffle=True,
                                      stratify=train_allFeatures_df['is_signal_channel'])
  t_input_df=t_allFeatures_df[inputFeatures]
  v_input_df=v_allFeatures_df[inputFeatures]
  test_input_df = test_allFeatures_df[inputFeatures]

  sc = StandardScaler().fit(allFeatures_df[inputFeatures])
  scaled_t_input_df=sc.transform(t_input_df)
  scaled_v_input_df=sc.transform(v_input_df)
  scaled_test_input_df=sc.transform(test_input_df)

  return allFeatures_df, scaled_t_input_df, t_target_df, scaled_v_input_df, v_target_df, scaled_test_input_df, test_target_df, test_allFeatures_df, target_df
  
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

def do_training(validation_frac, test_frac, batch_size, n_epochs, dropoutRate, nodes, trainingSample):

  ########################################################################
  ########################################################################
  # Preparation of the input featues
  ########################################################################
  ########################################################################
  
  plotsDir = 'plots/dnnFeedForward/'+ trainingSample + '_channel/'
  resultsDir = 'results/dnnFeedForward/' + trainingSample + '_channel/'
  
  predictionSample = trainingSample

  inputFeatures = [
    #'Bpt', 'Beta', 'Bphi',
    'pv_sv_dist',
    'Bpx', 'Bpy', 'Bpz',
    'jpsi_mass',
    'mu1pt', 'mu1eta', 'mu1phi',
    'mu2pt', 'mu2eta', 'mu2phi',
    'kpt', 'keta', 'kphi',
    #'jpsivtx_vtx_x', 'jpsivtx_vtx_y', 'jpsivtx_vtx_z', 
    #'jpsivtx_vtx_ex', 'jpsivtx_vtx_ey', 'jpsivtx_vtx_ez', 
    ]
  allFeatures_df, t_input_df, t_target_df, v_input_df, v_target_df, test_input_df, test_target_df, test_allFeatures_df, target_df = get_arrays(trainingSample=trainingSample, predictionSample=predictionSample, inputFeatures=inputFeatures)

  #target_df = pd.concat([t_target_df, v_target_df, test_target_df])
  target_np = target_df.to_numpy().reshape(-1, 1)
  sc_target = StandardScaler().fit(target_np)
  scaled_t_target_df = sc_target.transform(t_target_df.to_numpy().reshape(-1, 1))
  scaled_v_target_df = sc_target.transform(v_target_df.to_numpy().reshape(-1, 1))
  #######
  data_len=len(inputFeatures)
  K.clear_session()
  model = build_regression_model(nodes=nodes, dropoutRate=dropoutRate, data_len=data_len)
  history=model.fit(t_input_df , 
                    #scaled_t_target_df, 
                    t_target_df, 
                    epochs=n_epochs, verbose=1, batch_size=batch_size, 
                    #validation_data=(v_input_df , scaled_v_target_df),
                    validation_data=(v_input_df , v_target_df),
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience = 10, verbose=1),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1),
                        TerminateOnNaN()]
                    )
  
  # Model to json
  model_json = model.to_json()
  with open("model.json","w") as json_file:
    json_file.write(model_json)

  # training weights to HDF5
  model.save_weights("model_weights.h5")
  print("Model saved to disk.")



  history_df = pd.DataFrame(history.history, index=history.epoch)
  test_allFeatures_out_df = test_allFeatures_df.copy()
  '''
  bc_pt_scaled_regression = np.transpose(model.predict(test_input_df))
  bc_pt_regression = sc_target.inverse_transform(bc_pt_scaled_regression[0])
  '''

  bc_pt_regression_tmp = np.transpose(model.predict(test_input_df))
  bc_pt_regression = bc_pt_regression_tmp[0]
  test_allFeatures_out_df['mu1_grandmother_pt_regression'] = bc_pt_regression
  
  outputFile = 'output_test.root' 
  historyFile = 'history_test.root'

  inputTree = 'BTo3Mu'
  test_allFeatures_out_df.to_root(outputFile, key=inputTree)
  #results_df.to_root(outputFile, key='tree')
  history_df.to_root(historyFile, key='historyTree')

def do_prediction(loaded_model):
  loaded_model.compile(loss='mse', optimizer='adam',metrics=['mse', 'mae'])

  inputFeatures = [
    #'Bpt', 'Beta', 'Bphi',
    'pv_sv_dist',
    'Bpx', 'Bpy', 'Bpz',
    'jpsi_mass',
    'mu1pt', 'mu1eta', 'mu1phi',
    'mu2pt', 'mu2eta', 'mu2phi',
    'kpt', 'keta', 'kphi',
    #'jpsivtx_vtx_x', 'jpsivtx_vtx_y', 'jpsivtx_vtx_z', 
    #'jpsivtx_vtx_ex', 'jpsivtx_vtx_ey', 'jpsivtx_vtx_ez', 
    ]
  #inputFile = "/gpfs/ddn/srm/cms/store/user/garamire/ntuples/2021Mar23/BcToJPsiMuMu_is_jpsi_mu_merged.root"
  #inputFile = "data/BcToJPsiMuMu_is_jpsi_lepton_train.root"
  inputFile = "data/alldata_selected.root"
  inputTree = 'BTo3Mu'
  allFeatures_df = read_root(inputFile, inputTree)
  #target_df = allFeatures_df['mu1_grandmother_pt'].copy() #TODO: Change this to the k gen pt info once is included in the gen file

  test_input_df = allFeatures_df[inputFeatures]
  sc = StandardScaler().fit(test_input_df)
  scaled_input_df = sc.transform(test_input_df)


  '''
  target_np = target_df.to_numpy().reshape(-1, 1)
  sc_target = StandardScaler().fit(target_np)
  scaled_target_df = sc_target.transform(target_np.reshape(-1, 1))
  bc_pt_scaled_regression = np.transpose(loaded_model.predict(scaled_input_df))
  bc_pt_regression = sc_target.inverse_transform(bc_pt_scaled_regression[0])
  '''
  bc_pt_regression_tmp = np.transpose(loaded_model.predict(scaled_input_df))
  bc_pt_regression = bc_pt_regression_tmp[0]


  test_allFeatures_out_df = allFeatures_df.copy()
  test_allFeatures_out_df['mu1_grandmother_pt_regression'] = bc_pt_regression
  
  outputFile = 'output_test_read.root' 

  test_allFeatures_out_df.to_root(outputFile, key=inputTree)

def main():
  validation_frac, test_frac, batch_size, n_epochs, dropoutRate, nodes, trainingSample = get_parameters()
  force_training = True

  is_model_file_present = True
  model_file_name = "model.json"
  weights_file_name = "model_weights.h5"
  try: 
    json_file = open(model_file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    try:
      loaded_model.load_weights(weights_file_name)
      print("Loaded model from disk")
    except OSError:
      print("Could not open the file %s" % (weights_file_name))

  except OSError:
    print("Could not open the file %s" % (model_file_name))
    is_model_file_present = False


  if (force_training or not is_model_file_present):
    do_training(validation_frac, test_frac, batch_size, n_epochs, dropoutRate, nodes, trainingSample)
  else:
    do_prediction(loaded_model)

if __name__=="__main__":
    main()
