#!/usr/bin/env python
# vim: ts=2 sw=2 expandtab softtabstop=2 ai smarttab
import os, sys, math
import numpy as np
from ROOT import gPad, gStyle
from ROOT import TFile, TTree, TCanvas, TH1, TH1F, Math, TLorentzVector
from featuresList import featuresList
import copy
gStyle.SetOptStat(0)
muonPdgMass = 0.113428
jpsiPdgMass = 3.0969
bcPdgMass = 6.2756

def computeProcessedFeatures(muon1_p4, muon2_p4, unpairedMuon_p4, jpsi_p4, bc_p4):
  nProcessedFeatures = 16 # 10 processed features + 6 flags for channels and trigger categories
  if ( bc_p4.M() != 0): 
    bcPtCorrected = (bcPdgMass * bc_p4.Pt())/ bc_p4.M()
  else:
    bcPtCorrected = bc_p4.Pt()

  #muonsSystem_p4 = muon1_p4 + muon2_p4 + unpairedMuon_p4

  bcCorrected_p4 = TLorentzVector()
  bcCorrected_p4.SetPtEtaPhiM(bcPtCorrected,
      jpsi_p4.Eta(),
      jpsi_p4.Phi(),
      bc_p4.M())
  boostToBcCorrectedRestFrame = -bcCorrected_p4.BoostVector()
  #boostToJpsiRestFrame = -(muon1_p4+muon2_p4).BoostVector()
  boostToJpsiRestFrame = -jpsi_p4.BoostVector()

  unpairedMuonBoostedToBcCorrectedRestFrame_p4 = copy.copy(unpairedMuon_p4)
  unpairedMuonBoostedToBcCorrectedRestFrame_p4.Boost(boostToBcCorrectedRestFrame)
  unpairedMuonBoostedToJpsiRestFrame_p4 = copy.copy(unpairedMuon_p4)
  unpairedMuonBoostedToJpsiRestFrame_p4.Boost(boostToJpsiRestFrame)

  nn_energyBcRestFrame = unpairedMuonBoostedToBcCorrectedRestFrame_p4.E()
  #nn_missMass2 = (bcCorrected_p4 - muon1_p4 - muon2_p4 - unpairedMuon_p4).M2()
  nn_missMass2 = (bcCorrected_p4 - jpsi_p4 - unpairedMuon_p4).M2()
  #nn_q2 = (bcCorrected_p4 - muon1_p4 - muon2_p4).M2()
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
  
  featuresEntry = np.array([[nn_energyBcRestFrame],
    [nn_missMass2],
    [nn_q2],
    [nn_missPt],
    [nn_energyJpsiRestFrame],
    [nn_varPt],
    [nn_deltaRMu1Mu2],
    [nn_unpairedMuPhi],
    [nn_unpairedMuPt],
    [nn_unpairedMuEta]], dtype=np.float32)

  return featuresEntry
  


def bcSelector(event, isBkg = False):
  iBcSelected = []
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
    if((event.truthMatchMuNegative[iBc] < 1 or event.truthMatchMuPositive[iBc] < 1) and not isBkg): continue
    if(event.truthMatchUnpairedMu[iBc] < 1.0 and not isBkg): continue
    iBcSelected.append(iBc)
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
          event.Bc_jpsi_px[iBc],
          event.Bc_jpsi_pz[iBc],
          event.Bc_jpsi_mass[iBc])

      featuresEntry = computeProcessedFeatures(muon1_p4, muon2_p4, unpairedMuon_p4, jpsi_p4, bc_p4)
      featuresEntry = np.append(featuresEntry, np.array([[event.triggerMatchDimuon0[iBc]]]), axis=0)
      featuresEntry = np.append(featuresEntry, np.array([[event.triggerMatchJpsiTk[iBc]]]), axis=0)
      featuresEntry = np.append(featuresEntry, np.array([[event.triggerMatchJpsiTkTk[iBc]]]), axis=0)
      featuresEntry = np.append(featuresEntry, np.array([[event.signalDecayPresent[iBc]]]), axis=0)
      featuresEntry = np.append(featuresEntry, np.array([[event.normalizationDecayPresent[iBc]]]), axis=0)
      featuresEntry = np.append(featuresEntry, np.array([[event.background1DecayPresent[iBc]]]), axis=0)

      featuresProcessed = np.append(featuresProcessed,featuresEntry, axis=1)

  return featuresProcessed

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
        featuresEntry[ifeature,0] = event.__getattr__(featuresList[ifeature])[iBc]
    
      features = np.append(features,featuresEntry, axis=1)
  return features


def main():
  nFeatures = len(featuresList)
  nFeaturesProcessed = 16
  features = np.array([[]]* nFeatures, dtype=np.float32)
  featuresProcessed = np.array([[]]*nFeaturesProcessed, dtype=np.float32)

  channels=["tau", "muon", "jpsiX"]
  for channel in channels:
    print("Selecting "+channel+" channel events.")
    features = fillFeaturesArray(features, channel)
    featuresProcessed = fillProcessedFeaturesArray(featuresProcessed, channel)
  
  np.savez_compressed('featuresData.npz', features)
  np.savez_compressed('featuresProcessedData.npz', featuresProcessed)
  
  print("Done.")


if __name__ == "__main__":
  main()
