import os
import numpy as np
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot 


print("Reading the input file...")

f = np.load("featuresData.npz")
inputData = f["arr_0"]
nn_inputFeatures = inputData[0:10,:]
triggerFlags = (inputData[10:13,:]).astype(int)
channelFlags = (inputData[13:16,:]).astype(int) == 1

def plotPtVsEta(ptData, etaData, categories):
    for i in range(len(categories)):
        cat = categories[i,:]
        pyplot.hist2d(etaData[cat], ptData[cat], bins=50, range = [[-2.4,2.4], [0.,25.]])
        sufix = "OnlyDimuon0"
        if(i == 1): sufix = "OnlyJPsiTk"
        if(i == 2): sufix = "Dimuon0_AND_JPsiTk"
        #pyplot.legend()
        pyplot.xlabel("$\eta(\mu_{unpaired})$")
        pyplot.ylabel("$p_{T} [GeV]$")
        #pyplot.legend(0)
        pyplot.savefig("plots/h2DEtaVsPt_"+sufix+".png")

def plotHistoByCategories(data, categories, labels, colors, name = "name___", log = False, myBins = None):
    mean = np.mean(data)
    stdev = np.std(data)
    lenbin = stdev/(len(data)**0.5)

    if (lenbin == 0):
        return

    binning = np.arange(mean-2*stdev, mean + 2*stdev , lenbin*10)

    if myBins is not None:
        binning = mybins

    pyplot.clf()

    for i in range(len(categories)):
        #cat = np.multiply(categories[i,:],triggerFlags[2,:])
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

#categories = (inputData[13:16,:]).astype(int) == 1
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
plotPtVsEta(nn_inputFeatures[8,:], nn_inputFeatures[9,:], categoriesSignal)
for j in range(len(varNames)):
    plotHistoByCategories ( nn_inputFeatures[j,:], categoriesSignal, labels, colors, name = varNames[j] )


print('Done!')
