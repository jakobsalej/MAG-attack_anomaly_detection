from datetime import datetime
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data_analysis import DataAnalysis
from data_preparation import DataPreparation


np.set_printoptions(precision=4)


def plotResults(trainScores, testScores, metric, draw=False):
    # plot results (and save them to .csv)
    metricLabel = ''
    if metric == 'acc':
        metricLabel = 'Accuracy'
    elif metric == 'balanced_acc':
        metricLabel = 'Balanced Accuracy'

    train = pd.DataFrame(data=trainScores)
    test = pd.DataFrame(data=testScores)

    # save to .csv
    train.to_csv(f'{dirName}/results/train_scores_{metric}.csv')
    test.to_csv(f'{dirName}/results/test_scores_{metric}.csv')

    trainResults = train.melt(
        'Samples', var_name='Algorithm', value_name=metricLabel)
    testResults = test.melt(
        'Samples', var_name='Algorithm', value_name=metricLabel)

    # plot
    sns.set()
    # plt.ticklabel_format(style='plain', axis='y')

    # training accuracy
    plt.close('all')
    plt.figure()
    trainingPlot = sns.pointplot(x='Samples', y=metricLabel, hue='Algorithm',
                                 data=trainResults, legend=True, legend_out=True)

    # testing accuracy
    plt.figure()
    testingPlot = sns.pointplot(x='Samples', y=metricLabel, hue='Algorithm',
                                data=testResults, legend=True, legend_out=True)

    # save plot images
    trainingPlot.get_figure().savefig(f'{dirName}/graphs/training_scores_{metric}.png')
    testingPlot.get_figure().savefig(f'{dirName}/graphs/testing_scores_{metric}.png')

    # draw
    if draw:
        plt.show()


def createDir(name=datetime.now().strftime("%d-%m-%Y(%H-%M-%S)")):
    resFolderName = 'results/mode_0'
    try:
        # check if results folder exists already - if not, create it
        if not os.path.exists(f'../{resFolderName}'):
            os.mkdir(f'../{resFolderName}')

        # create new folder for results from this run
        newFolder = f'../{resFolderName}/{name}'
        os.mkdir(newFolder)
        os.mkdir(f'{newFolder}/results')
        os.mkdir(f'{newFolder}/graphs')
        os.mkdir(f'{newFolder}/datasets')
        return newFolder
    except FileExistsError:
        print("Directory ", name,  " already exists")
        return None


def saveDataset(data, fileName, folder='datasets'):
    try:
        if not os.path.exists(f'{dirName}/{folder}'):
            os.mkdir(f'{dirName}/{folder}')

        data.to_csv(f'{dirName}/{folder}/{fileName}.csv')
    except:
        print("Something went wrong")


def saveScoresToCSV(fullTrain, fullTest):
    for alg in fullTrain:
        fullTrainDF = pd.DataFrame(data=fullTrain[alg], index=[
            'accuracy', 'std', 'balanced_accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'])
        fullTestDF = pd.DataFrame(data=fullTest[alg], index=[
            'accuracy', 'balanced_accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'])

        fullTrainDF.to_csv(f'{dirName}/results/train_results_{alg}.csv')
        fullTestDF.to_csv(f'{dirName}/results/test_results_{alg}.csv')


def saveDataInfoToCSV(dataInfo):
    # map class indexes back to their labels
    normalityMapping = dp.getNormalityMapping()
    normalityClasses = [normalityMapping[index] for index in normalityMapping]
    # last row is sum of all classes
    normalityClasses.append('sum')
    dataInfoDF = pd.DataFrame(data=dataInfo, index=normalityClasses)
    # save to csv
    dataInfoDF.to_csv(f'{dirName}/datasets/data_info.csv')


def savePredictionScores(noOfSamples, predictions, datasetSize):
    # add no. of samples to dict for plotting
    trainScoresAcc['Samples'].append(noOfSamples)
    testScoresAcc['Samples'].append(noOfSamples)
    trainScoresBalancedAcc['Samples'].append(noOfSamples)
    testScoresBalancedAcc['Samples'].append(noOfSamples)

    for algName in predictions:
        (train, test) = predictions[algName]

        # add all scores for .csv
        if algName not in fullTrain:
            fullTrain[algName] = {}
            fullTest[algName] = {}

        fullTrain[algName][datasetSize] = train
        fullTest[algName][datasetSize] = test

        # add accuracy / balanced_accuracy of each algorithm for plotting
        # train: 0 = accuracy, 2 = balanced_accuracy
        # test: 0 = accuracy, 1 = balanced_accuracy
        if algName in trainScoresAcc:
            # accuracy
            trainScoresAcc[algName].append(train[0])
            testScoresAcc[algName].append(test[0])

            # balanced accuracy
            trainScoresBalancedAcc[algName].append(train[2])
            testScoresBalancedAcc[algName].append(test[1])
        else:
            # accuracy
            trainScoresAcc[algName] = [train[0]]
            testScoresAcc[algName] = [test[0]]

            # balanced accuracy
            trainScoresBalancedAcc[algName] = [train[2]]
            testScoresBalancedAcc[algName] = [test[1]]

def main():
    if not dirName:
        return -1

    da = DataAnalysis(dirName=dirName, mode=0, pi=PI)

    # Run predictions
    for datasetSize in (selectedSizes or [0.2, 0.4, 0.6, 0.8, 1]):
        # get the percentage of all data
        sampleData = dp.returnData(datasetSize, randomSeed=RANDOM_SEED)
        saveDataset(sampleData, f'AD_set_{datasetSize * 100:.0f}_seed{RANDOM_SEED}')

        # split data into X and y
        X, y = da.splitXY(sampleData)

        # split data into training (80%) and testing (20%) set
        xTrain, xTest, yTrain, yTest = da.splitTrainTest(
            X, y, trainSize=0.8, scale=True, resample=True, randomSeed=RANDOM_SEED)

        # save class distribution for this set
        da.saveClassDistribution(f'AD_set_{datasetSize * 100:.0f}_seed{RANDOM_SEED}', y)
        da.saveClassDistribution(f'AD_set_train{datasetSize * 100:.0f}_seed{RANDOM_SEED}', yTrain)
        da.saveClassDistribution(f'AD_set_test{datasetSize * 100:.0f}_seed{RANDOM_SEED}', yTest)

        # get data characteristics for current dataset size
        dataInfo[datasetSize] = da.getDataCharacteristics(yTrain, yTest)

        # get no. of samples and prediction accuracy (trainSize is set to 1, so no further data splitting is done)
        # multiple runs make no sense here, since we always take whole set
        noOfSamples, predictions = da.getScores(xTrain, xTest, yTrain, yTest, trainSize=datasetSize, randomSeeds=[RANDOM_SEED], mode=0, selectedAlgorithms=selectedAlgorithms) if selectedAlgorithms else da.getScores(
            xTrain, yTrain, xTest, yTest, trainSize=datasetSize, randomSeeds=[RANDOM_SEED], mode=0)

        # save results for graphs and .csv files
        savePredictionScores(noOfSamples, predictions, datasetSize)

    # save scores of all used algorithms to .csv
    saveScoresToCSV(fullTrain, fullTest)

    # save data info to .csv
    saveDataInfoToCSV(dataInfo)

    # save class distribution to csv
    da.getClassDistribution().to_csv(f'{dirName}/datasets/class_distribution.csv')

    # save plots (to also draw them, pass draw=True as param)
    plotResults(trainScoresAcc, testScoresAcc, metric='acc', draw=False)
    plotResults(trainScoresBalancedAcc, testScoresBalancedAcc, metric='balanced_acc', draw=False)



if __name__ == '__main__':
    # MODE 0:
    # take percentage of all data, then split it into train (80%) / test (20%)

    # to run on Raspberry Pi, set to True
    PI = True

    # select dataset sizes (up to 1.0) and algorithms (all options: 'logReg', 'svm', 'dt', 'rf', 'ann')
    selectedSizes = [0.2, 0.4, 0.6, 0.8, 1]
    selectedAlgorithms = ['logReg', 'svm', 'dt', 'rf', 'ann']
    #selectedAlgorithms = ['ann']

    # set random seed for data sampling
    RANDOM_SEED = 42

     # set to True if training set should be resampled for a more balanced set
    SHOULD_RESAMPLE = True

    # create new directory for results of this run
    # name of the folder can be passed as param (default name is timestamp)
    dirName = createDir()

    # init dicts to hold data
    dataInfo = {}
    fullTrain = {}
    fullTest = {}

    # scores for plotting
    trainScoresAcc = {'Samples': []}    
    testScoresAcc = {'Samples': []}
    trainScoresBalancedAcc = {'Samples': []}
    testScoresBalancedAcc = {'Samples': []}

    # clean and preprocess data
    dp = DataPreparation('../data/mainSimulationAccessTraces.csv')
    dp.prepareData()

    # run predictions
    main()
