import itertools
import matplotlib.pyplot as plt
import pandas as pd
import root_pandas
from root_pandas import read_root

trainingSample = 'mix'
resultsDir = 'results/dnnFeedForward/' + trainingSample + '_channel/'
plotsDir = 'plots/dnnFeedForward/' + trainingSample + '_channel/'
nNodesAllowed = [20, 40, 60, 80, 100]
minNumberOfLayers = 1
maxNumberOfLayers = 5

def plotHistory(history_df, historyFile):
    plt.figure(num=None, figsize=(5, 6), dpi = 300, facecolor='w', edgecolor='k')
    plt.legend(fontsize='x-small')
    ax = plt.subplot(211)
    history_df['loss'].plot(ax=ax, legend=1, label=historyFile)
    plt.title("Loss")
    ax = plt.subplot(212)
    history_df['mean_squared_error'].plot(ax=ax, legend=1, label=historyFile)
    plt.title("MSE")
    plt.xlabel("Epochs")
    plt.tight_layout()
    plt.savefig(plotsDir+historyFile+'_loss_and_mse_history.png')
    plt.close()
    pass

def plotRatio(input_df, inputFile):
    plt.figure(num=None, figsize=(5, 6), dpi = 300, facecolor='w', edgecolor='k')
    plt.legend(fontsize='x-small')
    input_df['bc_ptRatio_predictedGen'].hist( bins= 30, label=inputFile, histtype = 'step')
    plt.legend(fontsize='x-small')
    plt.xlabel("pt_predicted_nn/pt_gen")
    plt.tight_layout()
    plt.savefig(plotsDir+inputFile+'_ratio_predicted_gen.png')
    plt.close()




inputFile1 = read_root(resultsDir + "results-nodes_20.root", 'tree')
ratioCorrectedGen = inputFile1['bc_ptRatio_correctedGen']
print("%5.3f" % ratioCorrectedGen.std())

labelList = []
stdList = []
meanList = []
for L in range(minNumberOfLayers, maxNumberOfLayers):
    for nodes in itertools.product(nNodesAllowed, repeat=L):
        inputFile ="results-nodes"
        historyFile ="history-nodes"
        label = "nodes"
        for node in nodes:
            inputFile += "_%d" % node
            historyFile += "_%d" % node
            label += "_%d" % node
        labelList.append(label)
        input_df = read_root(resultsDir + inputFile +".root", 'tree')
        history_df = read_root(resultsDir+historyFile+".root", 'historyTree')
        plotHistory(history_df=history_df, historyFile=historyFile)
        plotRatio(input_df=input_df, inputFile=inputFile)
        
        stdList.append(input_df['bc_ptRatio_predictedGen'].std())
        meanList.append(input_df['bc_ptRatio_predictedGen'].mean())
        #print(inputFile)
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

print(std_df['nodes'][std_df['std'] < 0.138])

std_df['std'].hist(bins=20,
        color='DarkBlue',
        histtype='step')
plt.grid(False)
#plt.title("ratios standard deviation")
plt.xlabel("ratios standard deviation")
plt.savefig(plotsDir+'histoMeans.png')
plt.clf()


del std_df
