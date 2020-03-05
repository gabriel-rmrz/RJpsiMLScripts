#!/usr/bin/env python
# vim: ts=2 sw=2 expandtab softtabstop=2 ai smarttab
import os, sys, math
import numpy as np
from ROOT import gPad, gStyle
from ROOT import TFile, TTree, TCanvas, TH1, TH1F, TLorentzVector
from featuresList import allFeatures, processedFeatures

outputFeaturesList = processedFeatures
nFeatures = len(outputFeaturesList)
features_tmp0 = np.array([[]]* nFeatures, dtype=np.float32)


def computeProcessedFeatures(muon1_p4, muon2_p4, unpairedMuon_p4, jpsi_pt, bc_p4):
  processedVars = []



def bcSelector(inputTree, features_tmp, isBkg = False):
  featuresEntry = np.array([[0]]* nFeatures, dtype=np.float32)
  print(featuresEntry.shape)
  for event in inputTree:
    if(event.nBc < 1) : continue 
    for iBc in range(event.nBc):
      if((event.signalDecayPresent[iBc]<1) and (event.normalizationDecayPresent[iBc] < 1) and (event.background1DecayPresent[iBc] < 1) ) : continue
      if(event.triggerMatchDimuon0[iBc] < 1 and event.triggerMatchJpsiTk[iBc] < 1 and event.triggerMatchJpsiTkTk[iBc] < 1) : continue
      if(abs(event.Bc_jpsi_mu1_eta[iBc]) > 2.4): continue
      if((event.Bc_jpsi_mu2_eta[iBc] < -2.4) or (event.Bc_jpsi_mu2_eta[iBc] > 2.4) ):continue
      if((event.Bc_jpsi_mu1_pt[iBc] < 4.0) or (event.Bc_jpsi_mu2_pt[iBc] < 4.0) ): continue
      if((event.Bc_jpsi_mass[iBc] < 2.95) or (event.Bc_jpsi_mass[iBc] > 3.25)): continue
      if((event.isMuon1Soft[iBc] < 1) or (event.isMuon2Soft[iBc] < 1) ): continue
      if(event.Bc_jpsi_pt[iBc] < 8.0): continue
      if(event.Bc_mass[iBc] < 2.0 or event.Bc_mass[iBc] >6.4): continue

      #if(event.Bc_vertexProbability[iBc] < 0.1): continue
      #if(event.Bc_jpsi_vertexProbability[iBc] < 0.05): continue
      #if(event.nn_ptUnpairedMu[iBc] <2): continue

      #Preparint quadrivectors to calculate the processed variables
      
      computeProcessedFeatures(event)

      if((event.truthMatchMuNegative[iBc] < 1 or event.truthMatchMuPositive[iBc] < 1) and not isBkg): continue
      if(event.truthMatchUnpairedMu[iBc] < 1.0 and not isBkg): continue

      for ifeature in range(nFeatures ):
          featuresEntry[ifeature,0] = event.__getattr__(outputFeaturesList[ifeature])[iBc]
      
      features_tmp = np.append(features_tmp,featuresEntry, axis=1)


      break
  print(features_tmp.shape)
  return features_tmp

gStyle.SetOptStat(0)
signalFileName = "rootuples/RootupleBcTo3Mu_tauChannel.root"
signalInputFile = TFile.Open(signalFileName)
signalInputTree = signalInputFile.Get("rootuple/ntuple")

normalizationFileName = "rootuples/RootupleBcTo3Mu_muonChannel.root"
normalizationInputFile = TFile.Open(normalizationFileName)
normalizationInputTree = normalizationInputFile.Get("rootuple/ntuple")

backgroundFileName = "rootuples/RootupleBcTo3Mu_jpsiXChannel.root"
backgroundInputFile = TFile.Open(backgroundFileName)
backgroundInputTree = backgroundInputFile.Get("rootuple/ntuple")


print("Selecting signal channel events.")
features_tmp1 = bcSelector(signalInputTree, features_tmp0, isBkg = False)
print("Selecting normalization channel events.")
features_tmp2 = bcSelector(normalizationInputTree, features_tmp1, isBkg = False)
print("Selecting background channel events.")
features_tmp3 = bcSelector(backgroundInputTree, features_tmp2, isBkg = True)
#print("Selecting signal channel events.")
#features_tmp3 = bcSelector(signalInputTree, features_tmp0, isBkg = False)

features = features_tmp3
np.savez_compressed('featuresData.npz', features)

print("Done.")
