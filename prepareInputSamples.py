#!/usr/bin/env python
# vim: ts=2 sw=2 expandtab softtabstop=2 ai smarttab
import os, sys, math
import numpy as np
from ROOT import gPad, gStyle
from ROOT import TFile, TTree, TCanvas, TH1, TH1F, Math, TLorentzVector
from utils.featuresList import featuresList
import copy
gStyle.SetOptStat(0)
muonPdgMass = 0.1056
jpsiPdgMass = 3.0969
bcPdgMass = 6.2756
nProcessedFeatures = 20 # 5 corrected pt + 10 processed features + 5 4-vectors energies + 6 trigger and decay channel information

def computeProcessedFeatures(muon1_p4, muon2_p4, unpairedMuon_p4, jpsi_p4, bc_p4):
  if ( bc_p4.M() != 0): 
    bcPtCorrected = (bcPdgMass * bc_p4.Pt())/ bc_p4.M()
  else:
    bcPtCorrected = bc_p4.Pt()

  muonsSystem_p4 = muon1_p4 + muon2_p4 + unpairedMuon_p4

  bcCorrected_p4 = TLorentzVector()
  bcCorrected_p4.SetPtEtaPhiM(bcPtCorrected,
      muonsSystem_p4.Eta(),
      muonsSystem_p4.Phi(),
      #bc_p4.M())
      bcPdgMass)
  boostToBcCorrectedRestFrame = -bcCorrected_p4.BoostVector()
  #boostToJpsiRestFrame = -(muon1_p4+muon2_p4).BoostVector()
  boostToJpsiRestFrame = -jpsi_p4.BoostVector()

  unpairedMuonBoostedToBcCorrectedRestFrame_p4 = TLorentzVector() 
  unpairedMuonBoostedToBcCorrectedRestFrame_p4.SetPtEtaPhiM(unpairedMuon_p4.Pt(), unpairedMuon_p4.Eta(), unpairedMuon_p4.Phi(), unpairedMuon_p4.M())
  unpairedMuonBoostedToBcCorrectedRestFrame_p4.Boost(boostToBcCorrectedRestFrame)
  unpairedMuonBoostedToJpsiRestFrame_p4 = TLorentzVector() 
  unpairedMuonBoostedToJpsiRestFrame_p4.SetPtEtaPhiM(unpairedMuon_p4.Pt(), unpairedMuon_p4.Eta(), unpairedMuon_p4.Phi(), unpairedMuon_p4.M())
  unpairedMuonBoostedToJpsiRestFrame_p4.Boost(boostToJpsiRestFrame)

  nn_energyBcRestFrame = unpairedMuonBoostedToBcCorrectedRestFrame_p4.E()
  #nn_missMass2 = (bcCorrected_p4 - muon1_p4 - muon2_p4 - unpairedMuon_p4).M2()
  nn_missMass2 = (bcCorrected_p4 - jpsi_p4 - unpairedMuon_p4).M2()
  #nn_q2 = (bc_p4 - muon1_p4 - muon2_p4).M2()
  nn_q2 = (bcCorrected_p4 - jpsi_p4).M2()
  #nn_missPt = bcCorrected_p4.Pt() - muon1_p4.Pt() - muon2_p4.Pt() - unpairedMuon_p4.Pt()
  nn_missPt = bcCorrected_p4.Pt() - jpsi_p4.Pt() - unpairedMuon_p4.Pt()
  nn_energyJpsiRestFrame = unpairedMuonBoostedToJpsiRestFrame_p4.E()
  #nn_varPt = (muon1_p4 + muon2_p4).Pt() - unpairedMuon_p4.Pt()
  nn_varPt = jpsi_p4.Pt() - unpairedMuon_p4.Pt()
  nn_deltaRMu1Mu2 = muon1_p4.DeltaR(muon2_p4)
  nn_unpairedMuPhi = unpairedMuon_p4.Phi()
  nn_unpairedMuPt = unpairedMuon_p4.Pt()
  nn_unpairedMuEta = unpairedMuon_p4.Eta()

  featuresEntry = np.array([
    [bcCorrected_p4.Pt()], 
    [bcCorrected_p4.Px()], 
    [bcCorrected_p4.Py()], 
    [bcCorrected_p4.Pz()], 
    [bcCorrected_p4.E()], 
    [nn_energyBcRestFrame],
    [nn_missMass2],
    [nn_q2],
    [nn_missPt],
    [nn_energyJpsiRestFrame],
    [nn_varPt],
    [nn_deltaRMu1Mu2],
    [nn_unpairedMuPhi],
    [nn_unpairedMuPt],
    [nn_unpairedMuEta]], 
    dtype=np.double)

  return featuresEntry
  


def bcSelector(event, isBkg = False):
  iBcSelected = []
  iBcJpsiPts = []
  for iBc in range(event.nBc):
    if((event.signalDecayPresent[iBc]<1) and (event.normalizationDecayPresent[iBc] < 1)) : continue
    if(event.triggerMatchDimuon0[iBc] < 1 and event.triggerMatchJpsiTk[iBc] < 1 and event.triggerMatchJpsiTkTk[iBc] < 1) : continue
    if(abs(event.Bc_jpsi_mu1_eta[iBc]) > 2.4): continue
    if((event.Bc_jpsi_mu2_eta[iBc] < -2.4) or (event.Bc_jpsi_mu2_eta[iBc] > 2.4) ):continue
    if((event.Bc_jpsi_mu1_pt[iBc] < 4.0) or (event.Bc_jpsi_mu2_pt[iBc] < 4.0) ): continue
    if((event.Bc_jpsi_mass[iBc] < 2.95) or (event.Bc_jpsi_mass[iBc] > 3.25)): continue
    if((event.isMu1Soft[iBc] < 1) or (event.isMu2Soft[iBc] < 1) ): continue
    if(event.Bc_jpsi_pt[iBc] < 8.0): continue
    if(event.Bc_mass[iBc] < 2.0 or event.Bc_mass[iBc] >6.4): continue
    if(event.Bc_vertexProbability[iBc] < 0.01): continue
    if(event.jpsiVertexProbability[iBc] < 0.01): continue
    if((event.truthMatchMu2[iBc] < 1 or event.truthMatchMu1[iBc] < 1) and not isBkg): continue
    if(event.truthMatchMu[iBc] < 1.0 and not isBkg): continue
    iBcJpsiPts.append(event.Bc_jpsi_pt[iBc])
    #if( iBc > 0):
      #print("nEvetns: ", event.nBc)
      #print("jpsi_pt: ", event.Bc_jpsi_pt[iBc])
      #print("mu_pt: ", event.Bc_mu_pt[iBc])
  if(len(iBcJpsiPts) > 0):
    #print()
    #print("jpsi_pt max index: ", iBcJpsiPts.index(max(iBcJpsiPts)))
    iBcSelected.append(iBcJpsiPts.index(max(iBcJpsiPts)))
      

  return iBcSelected


def fillProcessedFeaturesArray(featuresProcessed, channel):
  inputFileName = "rootuples/RootupleBcTo3Mu_"+channel+"Channel.root"
  inputFile = TFile.Open(inputFileName)
  inputTree = inputFile.Get("rootuple/ntuple")
  for event in inputTree:
    if(event.nBc < 1) : continue 
    iBcSelected = bcSelector(event, isBkg = False)
    for iBc in iBcSelected:
      muon1_p4 = TLorentzVector()
      muon2_p4 = TLorentzVector()
      unpairedMuon_p4 = TLorentzVector()
      jpsi_p4 = TLorentzVector()
      bc_p4 = TLorentzVector()

      muon1_p4.SetXYZM(event.Bc_jpsi_mu1_px[iBc],
          event.Bc_jpsi_mu1_py[iBc],
          event.Bc_jpsi_mu1_pz[iBc],
          muonPdgMass)
      muon2_p4.SetXYZM(event.Bc_jpsi_mu2_px[iBc],
          event.Bc_jpsi_mu2_py[iBc],
          event.Bc_jpsi_mu2_pz[iBc],
          muonPdgMass)
      unpairedMuon_p4.SetXYZM(event.Bc_mu_px[iBc],
          event.Bc_mu_py[iBc],
          event.Bc_mu_pz[iBc],
          muonPdgMass)
      bc_p4.SetXYZM(event.Bc_px[iBc],
          event.Bc_py[iBc],
          event.Bc_pz[iBc],
          event.Bc_mass[iBc])
      jpsi_p4.SetXYZM(event.Bc_jpsi_px[iBc],
          event.Bc_jpsi_py[iBc],
          event.Bc_jpsi_pz[iBc],
          event.Bc_jpsi_mass[iBc])

      #print("---")
      #print("event.gen_mu_p4.Px(): ", event.gen_mu_p4.Px())
      #print("event.Bc_mu_px[iBc]: ", event.Bc_mu_px[iBc])
      featuresEntry = computeProcessedFeatures(event.gen_jpsi_mu1_p4, event.gen_jpsi_mu2_p4, event.gen_mu_p4, event.gen_jpsi_p4, event.gen_b_p4)
      #featuresEntry = computeProcessedFeatures(muon1_p4, muon2_p4, unpairedMuon_p4, jpsi_p4, bc_p4)
      featuresEntry = np.append(featuresEntry, np.array([[bc_p4.E()]]), axis=0)
      featuresEntry = np.append(featuresEntry, np.array([[jpsi_p4.E()]]), axis=0)
      featuresEntry = np.append(featuresEntry, np.array([[muon1_p4.E()]]), axis=0)
      featuresEntry = np.append(featuresEntry, np.array([[muon2_p4.E()]]), axis=0)
      featuresEntry = np.append(featuresEntry, np.array([[unpairedMuon_p4.E()]]), axis=0)


      #featuresEntry = computeProcessedFeatures(event.gen_muonPositive_p4, event.gen_muonNegative_p4, event.gen_unpairedMuon_p4, event.gen_jpsi_p4, event.gen_b_p4)
      #featuresEntry = np.append(featuresEntry, np.array([[event.triggerMatchDimuon0[iBc]]]), axis=0)
      #featuresEntry = np.append(featuresEntry, np.array([[event.triggerMatchJpsiTk[iBc]]]), axis=0)
      #featuresEntry = np.append(featuresEntry, np.array([[event.triggerMatchJpsiTkTk[iBc]]]), axis=0)
      #featuresEntry = np.append(featuresEntry, np.array([[event.signalDecayPresent[iBc]]]), axis=0)
      #featuresEntry = np.append(featuresEntry, np.array([[event.normalizationDecayPresent[iBc]]]), axis=0)
      #featuresEntry = np.append(featuresEntry, np.array([[event.background1DecayPresent[iBc]]]), axis=0)
      

      featuresProcessed = np.append(featuresProcessed,featuresEntry, axis=1)

  return featuresProcessed

def fillGenFeaturesArray(genFeatures, channel):
  inputFileName = "rootuples/RootupleBcTo3Mu_"+channel+"Channel.root"
  inputFile = TFile.Open(inputFileName)
  inputTree = inputFile.Get("rootuple/ntuple")
  for event in inputTree:
    if(event.nBc < 1) : continue 
    iBcSelected = bcSelector(event, isBkg = False)
    for iBc in iBcSelected:
      featuresEntry = np.array([
        [event.gen_b_p4.Pt()], [event.gen_b_p4.Px()], [event.gen_b_p4.Py()], [event.gen_b_p4.Pz()], [event.gen_b_p4.E()], [event.gen_b_p4.Eta()], [event.gen_b_p4.Phi()],
        [event.gen_jpsi_p4.Pt()], [event.gen_jpsi_p4.Px()], [event.gen_jpsi_p4.Py()], [event.gen_jpsi_p4.Pz()], [event.gen_jpsi_p4.E()], [event.gen_jpsi_p4.Eta()], [event.gen_jpsi_p4.Phi()],
        [event.gen_jpsi_mu1_p4.Pt()], [event.gen_jpsi_mu1_p4.Px()], [event.gen_jpsi_mu1_p4.Py()], [event.gen_jpsi_mu1_p4.Pz()], [event.gen_jpsi_mu1_p4.E()], [event.gen_jpsi_mu1_p4.Eta()], [event.gen_jpsi_mu1_p4.Phi()],
        [event.gen_jpsi_mu2_p4.Pt()], [event.gen_jpsi_mu2_p4.Px()], [event.gen_jpsi_mu2_p4.Py()], [event.gen_jpsi_mu2_p4.Pz()], [event.gen_jpsi_mu2_p4.E()], [event.gen_jpsi_mu2_p4.Eta()], [event.gen_jpsi_mu2_p4.Phi()],
        [event.gen_mu_p4.Pt()], [event.gen_mu_p4.Px()], [event.gen_mu_p4.Py()], [event.gen_mu_p4.Pz()], [event.gen_mu_p4.E()], [event.gen_mu_p4.Eta()], [event.gen_mu_p4.Phi()],
        [event.gen_munu_p4.Pt()], [event.gen_munu_p4.Px()], [event.gen_munu_p4.Py()], [event.gen_munu_p4.Pz()], [event.gen_munu_p4.E()], [event.gen_munu_p4.Eta()], [event.gen_munu_p4.Phi()],
        [event.gen_taunu1_p4.Pt()], [event.gen_taunu1_p4.Px()], [event.gen_taunu1_p4.Py()], [event.gen_taunu1_p4.Pz()], [event.gen_taunu1_p4.E()], [event.gen_taunu1_p4.Eta()], [event.gen_taunu1_p4.Phi()],
        [event.gen_taunu2_p4.Pt()], [event.gen_taunu2_p4.Px()], [event.gen_taunu2_p4.Py()], [event.gen_taunu2_p4.Pz()], [event.gen_taunu2_p4.E()], [event.gen_taunu2_p4.Eta()], [event.gen_taunu2_p4.Phi()]
        ])
      genFeatures = np.append(genFeatures, featuresEntry, axis = 1)


  return genFeatures


def fillFeaturesArray(features, channel):
  inputFileName = "rootuples/RootupleBcTo3Mu_"+channel+"Channel.root"
  inputFile = TFile.Open(inputFileName)
  inputTree = inputFile.Get("rootuple/ntuple")
  nFeatures = len(featuresList)
  for event in inputTree:
    if(event.nBc < 1) : continue 
    iBcSelected = bcSelector(event, isBkg = False)
    for iBc in iBcSelected:
      featuresEntry = np.array([[0]]* nFeatures, dtype=np.float32)
      for ifeature in range(nFeatures):
        #print(ifeature)
        #print(featuresList[ifeature])
        featuresEntry[ifeature,0] = event.__getattr__(featuresList[ifeature])[iBc]
    
      features = np.append(features,featuresEntry, axis=1)
  return features


def main():
  outDir = 'data/'
  nFeatures = len(featuresList) 
  nGenFeatures = 56
  features = np.array([[]]* nFeatures, dtype=np.double)
  genFeatures = np.array([[]]* nGenFeatures, dtype=np.double)
  featuresProcessed = np.array([[]]*nProcessedFeatures, dtype=np.double)

  #channels=["tau", "muon", "jpsiX"]
  channels=["tau", "muon"]
  #channels=["muon"]
  #channels=["tau"]
  for channel in channels:
    print("Selecting "+channel+" channel events.")
    features = fillFeaturesArray(features, channel)
    featuresProcessed = fillProcessedFeaturesArray(featuresProcessed, channel)
    genFeatures = fillGenFeaturesArray(genFeatures, channel)


  print("features shape: ", features.shape)
  print("features processed shape: ", featuresProcessed.shape)
  print("genfeatures processed: ", genFeatures.shape)
  features_tmp = np.append(featuresProcessed, features, axis=0)
  features = np.append(genFeatures, features_tmp, axis=0)
  print(features.shape)
  
  
  np.savez_compressed(outDir + 'featuresData.npz', features)
  #np.savez_compressed(outDir + 'featuresProcessedData.npz', featuresProcessed)
  
  print("Done.")


if __name__ == "__main__":
  main()
