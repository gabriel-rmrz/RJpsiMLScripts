import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import root_pandas
from root_pandas import read_root
from ROOT import TLorentzVector

trainingSample = 'muon'
resultsDir = 'results/dnnFeedForward/' + trainingSample + '_channel/'
plotsDir = 'plots/dnnFeedForward/' + trainingSample + '_channel/'
nNodesAllowed = [60, 80, 100, 140, 180]
#nNodesAllowed = [60, 80]
minNumberOfLayers = 1
maxNumberOfLayers = 5

muonPdgMass = 0.1056
jpsiPdgMass = 3.0969
bcPdgMass = 6.2756

def fillQ2Array(inputDF, varCat):
  q2array = np.array([[]], dtype=np.float32)
  for index, row in inputDF.iterrows():
    muon1_p4 = TLorentzVector()
    muon2_p4 = TLorentzVector()
    unpairedMuon_p4 = TLorentzVector()
    jpsi_p4 = TLorentzVector()
    bc_p4 = TLorentzVector()

    if(varCat == 0):
      muon1_p4.SetXYZM(row['gen_jpsi_mu1_px'],
          row['gen_jpsi_mu1_py'],
          row['gen_jpsi_mu1_pz'],
          muonPdgMass)
      muon2_p4.SetXYZM(row['gen_jpsi_mu2_px'],
          row['gen_jpsi_mu2_py'],
          row['gen_jpsi_mu2_pz'],
          muonPdgMass)
      unpairedMuon_p4.SetXYZM(row['gen_mu_px'],
          row['gen_mu_py'],
          row['gen_mu_pz'],
          muonPdgMass)
      bc_p4.SetXYZM(row['gen_b_px'],
          row['gen_b_py'],
          row['gen_b_pz'],
          bcPdgMass)
      jpsi_p4.SetXYZM(row['gen_jpsi_px'],
          row['gen_jpsi_py'],
          row['gen_jpsi_pz'],
          jpsiPdgMass)
    
    elif(varCat == 1):
      muon1_p4.SetXYZM(row['Bc_jpsi_mu1_px'],
          row['Bc_jpsi_mu1_py'],
          row['Bc_jpsi_mu1_pz'],
          muonPdgMass)
      muon2_p4.SetXYZM(row['Bc_jpsi_mu2_px'],
          row['Bc_jpsi_mu2_py'],
          row['Bc_jpsi_mu2_pz'],
          muonPdgMass)
      unpairedMuon_p4.SetXYZM(row['Bc_mu_px'],
          row['Bc_mu_py'],
          row['Bc_mu_pz'],
          muonPdgMass)
      bc_p4.SetXYZM(row['bc_px_predicted'],
          row['bc_py_predicted'],
          row['bc_pz_predicted'],
          bcPdgMass)
      jpsi_p4.SetXYZM(row['Bc_jpsi_px'],
          row['Bc_jpsi_py'],
          row['Bc_jpsi_pz'],
          row['Bc_jpsi_mass'])
    

    #q2Entry = computeProcessedFeatures(event.gen_jpsi_mu1_p4, event.gen_jpsi_mu2_p4, event.gen_mu_p4, event.gen_jpsi_p4, event.gen_b_p4)
    q2Entry = computeProcessedFeatures(muon1_p4, muon2_p4, unpairedMuon_p4, jpsi_p4, bc_p4)
    q2array = np.append(q2array, q2Entry, axis=1)

  return q2array

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

  #nn_q2 = (bc_p4 - muon1_p4 - muon2_p4).M2()
  #nn_q2 = (bcCorrected_p4 - jpsi_p4).M2()
  nn_q2 = (bc_p4 - jpsi_p4).M2()

  featuresEntry = np.array([
    [nn_q2]], 
    dtype=np.float32)

  return featuresEntry

def plotHistory(history_df, historyFile):
    plt.figure(num=None, figsize=(5, 6), dpi = 300, facecolor='w', edgecolor='k')
    plt.legend(fontsize='x-small')
    ax = plt.subplot(211)
    history_df['loss'].plot(ax=ax, legend=1, label="training")
    history_df['val_loss'].plot(ax=ax, legend=1, label="validation")
    plt.title("MSE-"+historyFile)
    ax = plt.subplot(212)
    history_df['mean_absolute_error'].plot(ax=ax, legend=1, label="training")
    history_df['val_mean_absolute_error'].plot(ax=ax, legend=1, label="validation" )
    plt.title("MAE")
    plt.xlabel("Epochs")
    plt.tight_layout()
    plt.savefig(plotsDir+historyFile+'_loss_and_mse_history.png')
    plt.close()
    pass

def plotRatio(input_df, inputFile):
    plt.figure(num=None, figsize=(5, 6), dpi = 300, facecolor='w', edgecolor='k')
    plt.legend(fontsize='x-small')
    input_df['bc_ptRatio_predictedGen'].hist( bins= 30, label=inputFile, histtype = 'step')
    input_df['bc_ptRatio_predictedGen'].hist( bins= 30, label=inputFile, histtype = 'step')
    plt.legend(fontsize='x-small')
    plt.xlabel("pt_predicted_nn/pt_gen")
    plt.tight_layout()
    plt.savefig(plotsDir+inputFile+'_ratio_predicted_gen.png')
    plt.close()




#inputFile1 = read_root(resultsDir + "results-nodes_20.root", 'tree')
#ratioCorrectedGen = inputFile1['bc_ptRatio_correctedGen']
#print("%5.3f" % ratioCorrectedGen.std())

labelList = []
stdList = []
meanList = []
for L in range(minNumberOfLayers, maxNumberOfLayers):
    for nodes in itertools.product(nNodesAllowed, repeat=L):
        inputFile ="results-"+ trainingSample + "_channel-nodes"
        historyFile ="history-"+ trainingSample + "_channel-nodes"
        label = "nodes"
        for node in nodes:
            inputFile += "_%d" % node
            historyFile += "_%d" % node
            label += "_%d" % node
        inputFile += "-vfrac_0p3-dropoutrate_0p1-bsize_200"
        historyFile += "-vfrac_0p3-dropoutrate_0p1-bsize_200"
        labelList.append(label)
        input_df = read_root(resultsDir + inputFile +".root", 'tree')
        history_df = read_root(resultsDir+historyFile+".root", 'historyTree')
        #plotHistory(history_df=history_df, historyFile=historyFile)
        #plotRatio(input_df=input_df, inputFile=inputFile)
        #stdList.append(input_df['bc_ptRatio_predictedGen'].std())
        #meanList.append(input_df['bc_ptRatio_predictedGen'].mean())
        q2_gen_cut = 3.2

        q2_predicted = (input_df['nn_q2_predicted'][input_df['nn_q2'] > q2_gen_cut]).to_numpy()
        q2_gen = (input_df['nn_q2'][input_df['nn_q2'] > q2_gen_cut]).to_numpy()  # generation level
        # Variables categories: 0:gen, 1:Predicted, 2:CorrectedJona
        #varCatPredicted = 1
        #varCatGen = 0
        #q2_predicted = fillQ2Array(input_df, varCatPredicted)
        #q2_gen = fillQ2Array(input_df, varCatGen)

        q2_resolution = (q2_predicted - q2_gen)/q2_gen
        stdList.append(np.std(q2_resolution))
        meanList.append(np.mean(q2_resolution))
        #print(label)
        #print(np.mean(q2_resolution))
        #print('reso', q2_resolution)
        #print('predicted', q2_predicted)
        #print('gen', q2_gen)

        
std_df = pd.DataFrame (list(zip(labelList, stdList, meanList)), columns=['nodes','std', 'mean'])
plt.figure(num=None, figsize=(6, 6), dpi = 300, facecolor='w', edgecolor='k') 
plt.legend(fontsize='x-small')
std_df.plot.scatter(x='mean',
        y='std',
        c='DarkBlue')
plt.title("")
plt.xlabel("q^2 resolution mean")
plt.ylabel("q^2 resolution standard deviation")
plt.tight_layout()
plt.savefig(plotsDir+'stdsVsMeans.png')
plt.clf()

#print(std_df['nodes'][(abs(std_df['mean']) < .05) & (std_df['std'] < 0.14)])
print(std_df[['nodes','std']][abs(std_df['mean']+0.05) < .001])
print(std_df[['std']][abs(std_df['mean']+0.05) < .001].min())

std_df['std'].hist(bins=20,
        color='DarkBlue',
        histtype='step')
plt.grid(False)
#plt.title("ratios standard deviation")
plt.xlabel("q^2 resolution standard deviation")
plt.savefig(plotsDir+'histoMeans.png')
plt.clf()


del std_df
