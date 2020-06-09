import os
import itertools
import subprocess
import time

def main():
    nNodesAllowed = [20, 40, 60, 80, 100]
    
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
    
    for L in range(minNumberOfLayers, maxNumberOfLayers):
        #for nodes in itertools.combinations_with_replacement(nNodesAllowed, L):
        for nodes in itertools.product(nNodesAllowed, repeat=L):
            command = "python feedForwardRegression.py --trainingSample " + trainingSample + " --nodes "
            for node in nodes:
                command += "%d " % node
            print(command)
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

if __name__ == '__main__':
    main()
