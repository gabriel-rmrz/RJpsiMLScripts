import itertools
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import root_pandas

from root_pandas import read_root
from root_numpy import fill_hist
from ROOT import TH2F, TH1F, TCanvas, gStyle, gPad
from sklearn.metrics import mean_squared_error
import ROOT

trainingSample = 'mix'
resultsDir = 'results/dnnFeedForward/' + trainingSample + '_channel/'
plotsDir = 'plots/dnnFeedForward/' + trainingSample + '_channel/'

def plotProfile(input_df):
    gStyle.SetOptStat(0)
    hPtGenVsPtRatioPredictedGen = TH2F('hPtGenVsPtRatioPredictedGen', 'NN prediction', 80, 0., 80., 40, 0., 2.)
    fill_hist(hPtGenVsPtRatioPredictedGen, input_df[['gen_b_pt', 'bc_ptRatio_predictedGen']].to_numpy())
    profilePtGenVsPtRatioPredictedGen = hPtGenVsPtRatioPredictedGen.ProfileX()
    profilePtGenVsPtRatioPredictedGen.SetMarkerStyle(ROOT.kFullCircle)

    hPtGenVsPtRatioCorrectedGen = TH2F('hPtGenVsPtRatioCorrectedGen', 'Jonas correction', 80, 0., 80., 40, 0., 2.)
    fill_hist(hPtGenVsPtRatioCorrectedGen, input_df[['gen_b_pt', 'bc_ptRatio_correctedGen']].to_numpy())
    profilePtGenVsPtRatioCorrectedGen = hPtGenVsPtRatioCorrectedGen.ProfileX()
    profilePtGenVsPtRatioCorrectedGen.SetMarkerStyle(ROOT.kFullSquare)

    c1 = TCanvas('c1', 'c1', 700, 500)
    profilePtGenVsPtRatioPredictedGen.SetLineColor(ROOT.kAzure)
    profilePtGenVsPtRatioCorrectedGen.SetLineColor(ROOT.kOrange)
    #profilePtGenVsPtRatioPredictedGen.SetTitle("")
    gStyle.SetOptTitle(0)
    profilePtGenVsPtRatioPredictedGen.GetXaxis().SetTitle("pT_{gen}(Bc) [GeV]")
    profilePtGenVsPtRatioPredictedGen.GetYaxis().SetTitle("pT_{corrected}(B_{c}^{+})/pT_{gen}(B_{c}^{+})")
    profilePtGenVsPtRatioPredictedGen.Draw("")
    profilePtGenVsPtRatioCorrectedGen.Draw('same')
    gPad.BuildLegend()
    c1.SaveAs(plotsDir + 'profile.png')
    return 0

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

def plotRatio(input_df, inputFile):
    gStyle.SetOptStat(0)
    #hPtRatioPredicted = TH1F('hPtRatioPredicted', 'NN prediction, RMS=0.135', 40, 0., 2.)
    #hPtRatioCorrected = TH1F('hPtRatioPredicted', 'Colinear approximation, RMS=0.175', 40, 0., 2.)
    rmse1 = ((input_df['bc_ptRatio_predictedGen'].mean() - input_df['bc_ptRatio_predictedGen'])**2).mean()**.5
    rmse2 = ((input_df['bc_ptRatio_correctedGen'].mean() - input_df['bc_ptRatio_correctedGen'])**2).mean()**.5

    label1 = "NN prediction, RMS=%3.3f" % rmse1
    label2 = "Colinear correction, RMS=%3.3f" % rmse2
    hPtRatioPredicted = TH1F('hPtRatioPredicted', label1, 40, 0., 2.)
    hPtRatioCorrected = TH1F('hPtRatioPredicted', label2, 40, 0., 2.)

    fill_hist(hPtRatioPredicted, input_df['bc_ptRatio_predictedGen'].to_numpy())
    fill_hist(hPtRatioCorrected, input_df['bc_ptRatio_correctedGen'].to_numpy())

    c1 = TCanvas('c1', 'c1', 700, 500)
    hPtRatioPredicted.SetLineColor(ROOT.kAzure)
    hPtRatioCorrected.SetLineColor(ROOT.kRed)
    #profilePtGenVsPtRatioPredictedGen.SetTitle("")
    gStyle.SetOptTitle(0)
    hPtRatioPredicted.GetYaxis().SetTitle("Events/50 MeV")
    hPtRatioPredicted.GetXaxis().SetTitle("pT_{corrected}(B_{c}^{+})/pT_{gen}(B_{c}^{+})")
    hPtRatioPredicted.Draw("")
    hPtRatioCorrected.Draw('same')
    gPad.BuildLegend()
    c1.SaveAs(plotsDir + inputFile + '_ratio_predicted_gen.pdf')
    
    #plt.figure(num=None, figsize=(6, 5), dpi = 300, facecolor='w', edgecolor='k')
    #plt.legend(fontsize='x-small')
    #binning = np.arange(0.5, 2, 0.05)
    #input_df['bc_ptRatio_predictedGen'].hist( bins= binning, label="NN prediction, RMS=%3.3f" % rmse1, histtype = 'step', density=1, grid=False)
    #input_df['bc_ptRatio_correctedGen'].hist( bins= binning, label="Jona's correction, RMS=%3.3f" % rmse2, histtype = 'step', density=1, grid=False)
    #print(rmse1)
    #print(rmse2)

    #plt.legend()
    #plt.xlabel("pT_corrected(Bc)/pT_gen(Bc)")
    #plt.ylabel("a.u.")
    #plt.tight_layout()
    #plt.savefig(plotsDir+inputFile+'_ratio_predicted_gen.png')
    #plt.close()







#inputFile1 = read_root(resultsDir + "results-nodes_20.root", 'tree')
#ratioCorrectedGen = inputFile1['bc_ptRatio_correctedGen']
#print("%5.3f" % ratioCorrectedGen.std())

labelList = []
stdList = []
meanList = []

nodes = [100, 100, 100, 60]
valFractions = [0.4]
#dropoutRates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
dropoutRates = [0.1]
#batchSizes = [25, 50, 75, 100, 125, 150, 175, 200]
batchSizes = [125]

#valFractions = [0.1]
#dropoutRates = [0.1]
#batchSizes = [25]
for vf in valFractions:
    for dr in dropoutRates:
        for bs in batchSizes:
            inputFile ="results-"+ trainingSample + "_channel-nodes"
            historyFile ="history-"+ trainingSample + "_channel-nodes"
            label = "params"
            for node in nodes:
                inputFile += "_%d" % node
                historyFile += "_%d" % node
                #label += "_%d" % node
            inputFile += "-vfrac_0p%d-dropoutrate_0p%d-bsize_%d" % ((int)(10*vf), (int)(10*dr), bs)
            historyFile += "-vfrac_0p%d-dropoutrate_0p%d-bsize_%d" % ((int)(10*vf), (int)(10*dr), bs)
            label += "-vfrac_0p%d-dropoutrate_0p%d-bsize_%d" % ((int)(10*vf), (int)(10*dr), bs)

            labelList.append(label)
            input_df = read_root(resultsDir + inputFile +".root", 'tree')
            history_df = read_root(resultsDir+historyFile+".root", 'historyTree')
            plotHistory(history_df=history_df, historyFile=historyFile)
            plotRatio(input_df=input_df, inputFile=inputFile)
            plotProfile(input_df=input_df)

            standardDev = input_df['bc_ptRatio_predictedGen'].std()
            meanVal = input_df['bc_ptRatio_predictedGen'].mean()
            minVal = meanVal - 4*standardDev
            maxVal = meanVal + 4*standardDev
            stdList.append(standardDev)
            meanList.append(meanVal)

            #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #    print(input_df[(input_df['bc_ptRatio_predictedGen'] < minVal) | (input_df['bc_ptRatio_predictedGen'] > maxVal)])
        
std_df = pd.DataFrame (list(zip(labelList, stdList, meanList)), columns=['nodes','std', 'mean'])
plt.figure(num=None, figsize=(6, 6), dpi = 300, facecolor='w', edgecolor='k') 
plt.legend(fontsize='x-small')
std_df.plot.scatter(x='mean',
        y='std',
        c='DarkBlue')
plt.title("")
plt.xlabel("ratios mean")
plt.ylabel("ratios standard deviation")
plt.tight_layout()
plt.savefig(plotsDir+'ratiosVsMeans.png')
plt.clf()

print(std_df['nodes'][(std_df['mean'] > 0.995) & (std_df['mean'] < 1.005) & (std_df['std'] < 0.13)])

std_df['std'].hist(bins=20,
        color='DarkBlue',
        histtype='step')
plt.grid(False)
#plt.title("ratios standard deviation")
plt.xlabel("ratios standard deviation")
plt.savefig(plotsDir+'histoMeans.png')
plt.clf()


del std_df
