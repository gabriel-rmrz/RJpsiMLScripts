import os
import numpy as np
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot 

pyplot.figure(num=None, figsize=(8,8), dpi = 300, facecolor='w', edgecolor='k')

print("Reading the input file...")

f = np.load("featuresProcessedData.npz")
inputData = f["arr_0"]
nn_inputFeatures = inputData[0:10,:]
triggerFlags = (inputData[10:13,:]).astype(int)
channelFlags = (inputData[13:16,:]).astype(int) == 1

def plotPtVsEta(ptData, etaData, categories):
    pyplot.scatter(etaData[categories[0,]], ptData[categories[0,]], c='tab:blue', alpha=0.3, edgecolors = 'none', label='OnlyDimuon0' )
    pyplot.scatter(etaData[categories[1,]], ptData[categories[1,]], c='tab:orange', alpha=0.3, edgecolors = 'none', label='OnlyJpsiTk' )
    pyplot.scatter(etaData[categories[2,]], ptData[categories[2,]], c='tab:green', alpha=0.3, edgecolors = 'none', label='Dimuon0_and_JPsiTk' )
    pyplot.legend()
    pyplot.xlabel("$\eta(\mu_{unpaired})$")
    pyplot.ylabel("$p_{T} [GeV]$")
    pyplot.ylim(0.,15)
    pyplot.xlim(-2.5,2.5)
    pyplot.grid(True)
    pyplot.savefig("plots/h2DEtaVsPt.png")

def plotHistoByCategories(data, categories, labels, colors, name = "name___", log = False, myBins = None):
    mean = np.mean(data)
    stdev = np.std(data)
    lenbin = stdev/(len(data)**0.5)

    if (lenbin == 0):
        return

    nStdevs = 4
    binning = np.arange(mean-nStdevs*stdev, mean + nStdevs*stdev , lenbin*10)

    if myBins is not None:
        binning = mybins

    pyplot.clf()

    for i in range(len(categories)):
        cat = categories[i,:]
        color = colors[i]
        label = labels[i]

        pyplot.hist(data[cat], color = color, label = label, histtype = 'step', bins = binning, log = log, density = True)
        
        print(sum(cat))

    pyplot.legend()
    pyplot.xlabel(name)
    pyplot.ylabel("entries")
    pyplot.savefig("plots/"+name+".png")
    



# Input for the plots
intersection  = (np.logical_and(triggerFlags[0,:],triggerFlags[1,:])) == 1
onlyDimuon0 = (triggerFlags[0,:] - intersection) == 1
onlyJpsiTrack = (triggerFlags[1,:] - intersection) == 1
categories = np.array([onlyDimuon0,onlyJpsiTrack, intersection ])
categoriesSignal = np.array([np.logical_and(onlyDimuon0,channelFlags[0,:]), np.logical_and(onlyJpsiTrack,channelFlags[0,:]), np.logical_and(intersection,channelFlags[0,:]) ])
categoriesNormalization = np.array([np.logical_and(onlyDimuon0,channelFlags[1,:]), np.logical_and(onlyJpsiTrack,channelFlags[1,:]), np.logical_and(intersection,channelFlags[1,:]) ])

colors=["red", "black", "green"]
labels = ["Only Dimuon0", "Only Jpsi+Trk",  "Intersection"]

varNames = ["energyBcRestFrame",
        "missMass2",
        "q2",
        "missPt",
        "energyJpsiRestFrame",
        "varPt",
        "deltaRMu1Mu2",
        "phiUnpairedMu",
        "ptUnpairedMu",
        "etaUnpairedMu"]
plotPtVsEta(nn_inputFeatures[8,:], nn_inputFeatures[9,:], categories)
for j in range(len(varNames)):
    plotHistoByCategories ( nn_inputFeatures[j,:], categories, labels, colors, name = varNames[j] )


print('Done!')
