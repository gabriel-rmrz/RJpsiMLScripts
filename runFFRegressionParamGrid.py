import os
import itertools
import subprocess
import time

def main():
    nodes = [140, 60]
    valFractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    dropoutRates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    batchSizes = [25, 50, 75, 100, 125, 150, 175, 200]
    
    trainingSample = "muon"
    plotsDir = "plots/dnnFeedForward/" + trainingSample + "_channel"
    resultsDir = "results/dnnFeedForward/" + trainingSample + "_channel"
    #if( !path.exit(plotsDir)):
    try:
        os.makedirs(plotsDir)
    except:
        print('The plots directory already exist')

    try:
        os.makedirs(resultsDir)
    except:
        print('The results directory already exist')

    minNumberOfLayers = 1
    maxNumberOfLayers = 5
    
    for vf in valFractions:
        for dr in dropoutRates:
            for bs in batchSizes:
                command = "python feedForwardRegression.py --trainingSample " + trainingSample + " --nodes "
                for node in nodes:
                    command += "%d " % node
                command += " --valFrac %2.2f --dropoutRate %2.2f --batchSize %d" % (vf, dr, bs)

                print(command)
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

if __name__ == '__main__':
    main()
