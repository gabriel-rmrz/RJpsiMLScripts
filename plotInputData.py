import os
import numpy as np
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot 
from utils.featuresList import allFeaturesList
from sklearn.preprocessing import StandardScaler

pyplot.figure(num=None, figsize=(8,8), dpi = 300, facecolor='w', edgecolor='k')

print("Reading the input file...")

f = np.load("data/featuresData.npz")
inputData = f["arr_0"]
nn_inputFeatures = (inputData[61:71,:]).astype(float)
allEnergies = inputData[71:76,:]
triggerFlags = (inputData[155:157,:]).astype(int)
channelFlags = (inputData[175:177,:]).astype(int) == 1

def plotPtVsEta(ptData, etaData, categories):
    pyplot.scatter(etaData[categories[0,]], ptData[categories[0,]], c='tab:blue', alpha=0.3, edgecolors= 'none', label='OnlyDimuon0' )
    pyplot.scatter(etaData[categories[1,]], ptData[categories[1,]], c='tab:orange', alpha=0.3, edgecolors = 'none', label='OnlyJpsiTk' )
    pyplot.scatter(etaData[categories[2,]], ptData[categories[2,]], c='tab:green', alpha=0.3, edgecolors = 'none', label='Dimuon0_and_JPsiTk' )
    pyplot.legend()
    pyplot.xlabel("$\eta(\mu_{unpaired})$")
    pyplot.ylabel("$p_{T} [GeV]$")
    pyplot.ylim(0.,15)
    pyplot.xlim(-2.5,2.5)
    pyplot.grid(True)
    pyplot.savefig("plots/h2DEtaVsPt.png")
    pyplot.clf()

def plotAllEnergies(allEnergies):
    mean = np.mean(allEnergies[0,:])
    stdev = np.std(allEnergies[0,:])
    lenbin = stdev/(len(allEnergies[0,:])**0.5)
    nStdevs = 4
    binning = np.arange(mean-nStdevs*stdev, mean + nStdevs*stdev , lenbin*10)
    color = ['blue', 'red', 'green', 'black', 'orange']
    label = ['Bc', 'JPsi', 'Muon1', 'Muon2', 'UnpairedMuon']
    for i in range(5):
        pyplot.hist(allEnergies[i,:], color = color[i], label = label[i], bins = binning,  histtype = 'step', density = True)
    pyplot.legend()
    pyplot.xlabel("Energy [GeV]")
    #pyplot.ylim(0.,15)
    #pyplot.grid(True)
    pyplot.savefig("plots/allEnergies.png")
    pyplot.clf()

def plotHistoByCategories(data, categories, labels, colors, name = "name___", log = False, myBins = None, prefix=None):

    mean = np.mean(data)
    stdev = np.std(data)
    lenbin = stdev/(len(data)**0.5)

    if (lenbin == 0):
        return

    nStdevs = 4
    print(name)
    print(mean)
    print(stdev)
    print(lenbin)
    print(data.shape)
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
    pyplot.savefig("plots/"+prefix+"/"+prefix+"_"+name+".png")
    



# Input for the plots
intersection  = (np.logical_and(triggerFlags[0,:],triggerFlags[1,:])) == 1
onlyDimuon0 = (triggerFlags[0,:] - intersection) == 1
onlyJpsiTrack = (triggerFlags[1,:] - intersection) == 1
categories = np.array([onlyDimuon0,onlyJpsiTrack, intersection ])
categoriesSignal = np.array([np.logical_and(onlyDimuon0,channelFlags[0,:]), np.logical_and(onlyJpsiTrack,channelFlags[0,:]), np.logical_and(intersection,channelFlags[0,:]) ])
categoriesNormalization = np.array([np.logical_and(onlyDimuon0,channelFlags[1,:]), np.logical_and(onlyJpsiTrack,channelFlags[1,:]), np.logical_and(intersection,channelFlags[1,:]) ])

colors=["red", "black", "green"]
labels = ["Only Dimuon0", "Only Jpsi+Trk",  "Intersection"]

channelColors = ["orange", "blue"]
channelLabels = ["Signal", "Normalization"]

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
allFeaturesList = np.array(allFeaturesList)
inputFeaturesNumbers = [ 
  71, 72, 73, 74, 75, 
  76, 77, 78, 79, 80, 81, 82, 83, 84, 
  86, 87, 88, 89, 90, 91, 92, 93, 94, 
  101, 102, 103, 104, 105,
  111, 112, 113, 114,
  121, 122, 123, 124, 125,
  131, 132, 133, 134,
  138, 139, 140, 141]

inputFeatures = inputData[inputFeaturesNumbers,:]
inputFeaturesLabels = allFeaturesList[inputFeaturesNumbers]

sc = StandardScaler().fit(inputFeatures)
scaled_inputFeatures = sc.transform(inputFeatures)

plotPtVsEta(nn_inputFeatures[8,:], nn_inputFeatures[9,:], categories)
plotAllEnergies(allEnergies)
#for j in range(len(varNames)):
#    plotHistoByCategories ( nn_inputFeatures[j,:], categories, labels, colors, name = varNames[j] , prefix = "trigCategories")
#    plotHistoByCategories ( nn_inputFeatures[j,:], channelFlags, channelLabels, channelColors, name = varNames[j] , prefix = "channels")
for k in range(len(inputFeaturesLabels)):
#    plotHistoByCategories ( inputFeatures[k,:], categories, labels, colors, name = inputFeaturesLabels[k] , prefix = "trigCategories")
#    plotHistoByCategories ( inputFeatures[k,:], channelFlags, channelLabels, channelColors, name = inputFeaturesLabels[k] , prefix = "channels")
    plotHistoByCategories ( scaled_inputFeatures[k,:], channelFlags, channelLabels, channelColors, name = inputFeaturesLabels[k] , prefix = "scaled_channels")


print('Done!')
