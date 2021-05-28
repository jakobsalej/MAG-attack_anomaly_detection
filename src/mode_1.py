from data_preparation import DataPreparation
from data_analysis import DataAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')


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
    trainingPlot.get_figure().savefig(
        f'{dirName}/graphs/training_scores_{metric}.png')
    testingPlot.get_figure().savefig(
        f'{dirName}/graphs/testing_scores_{metric}.png')

    # draw
    if draw:
        plt.show()


def createDir(name=datetime.now().strftime("%d-%m-%Y(%H-%M-%S)")):
    resFolderName = 'results/mode_1'
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
        os.mkdir(f'{newFolder}/info')
        return newFolder
    except FileExistsError:
        print(f'Directory {name} already exists')
        return None


def saveDataset(data, fileName, folder='datasets'):
    try:
        if not os.path.exists(f'{dirName}/{folder}'):
            os.mkdir(f'{dirName}/{folder}')

        data.to_csv(f'{dirName}/{folder}/{fileName}.csv', index=False)
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
    dataInfoDF.to_csv(f'{dirName}/info/data_info.csv')


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

    da = DataAnalysis(dirName=dirName, mode=1, pi=PI)

    # use all available data
    sampleData = dp.returnData(1, randomSeed=randomSeeds[0])
    saveDataset(sampleData, 'AD_dataset')

    # split data into X and y
    X, y = da.splitXY(sampleData)

    # save class distribution for this set
    da.saveClassDistribution('AD_dataset', y)

    # split data into training (80%) and testing (20%) set
    xTrain, xTest, yTrain, yTest = da.splitTrainTest(
        X, y, trainSize=0.8, scale=True, resample=SHOULD_RESAMPLE, randomSeed=randomSeeds[0])

    # print number of testing samples
    print('### Number of test samples:', xTest.shape[0])

    saveDataset(xTrain.assign(normality=yTrain.values), 'AD_set_train')
    saveDataset(xTest.assign(normality=yTest.values), 'AD_set_test')

    # get class distribution
    da.saveClassDistribution('AD_set_train', yTrain)
    da.saveClassDistribution('AD_set_test', yTest)

    # get data characteristics for current dataset size
    dataInfo[1] = da.getDataCharacteristics(yTrain, yTest)

    # Run predictions
    for datasetSize in selectedSizes:
        # Custom mapping
        TRAIN_SETS = {
		    0.001: 'AD_subset_balanced_0.1.csv',
		    0.002: 'AD_subset_balanced_0.2.csv',
		    0.005: 'AD_subset_balanced_0.5.csv',
		    0.01: 'AD_subset_balanced_1.csv',
		    0.02: 'AD_subset_balanced_2.csv',
		    0.05: 'AD_subset_balanced_5.csv',
		    0.1: 'AD_subset_balanced_10.csv',
		    0.15: 'AD_subset_balanced_15.csv',
		    0.2: 'AD_subset_balanced_20.csv',
        }
        trainSetPath = f'../data/AD_datoteke/C7_random/{TRAIN_SETS[datasetSize]}'

        # TRAIN_SETS = {
		#     0.001: 'AD_set_train_reduced_0.001_0.001.csv',
		#     0.002: 'AD_set_train_reduced_0.002_0.001.csv',
		#     0.005: 'AD_set_train_reduced_0.005_0.001.csv',
		#     0.01: 'AD_set_train_reduced_0.01_0.001.csv',
		#     0.02: 'AD_set_train_reduced_0.02_0.001.csv',
		#     0.05: 'AD_set_train_reduced_0.05_0.001.csv',
		#     0.1: 'AD_set_train_reduced_0.1_0.001.csv',
		#     0.15: 'AD_set_train_reduced_0.15_0.001.csv',
		#     0.2: 'AD_set_train_reduced_0.2_0.001.csv',
        # }
        # trainSetPath = f'../data/AD_datoteke/Class_cluster/{TRAIN_SETS[datasetSize]}'
        
        # print('file', trainSetPath)
        trainSetPath = None

        # get no. of samples and prediction accuracy
        noOfSamples, predictions = da.getScores(xTrain, xTest, yTrain, yTest, trainSetPath, trainSize=datasetSize, randomSeeds=randomSeeds, selectedAlgorithms=selectedAlgorithms)

        # save results for graphs and .csv files
        savePredictionScores(noOfSamples, predictions, datasetSize)

    # save scores of all used algorithms to .csv
    saveScoresToCSV(fullTrain, fullTest)

    # save data info to .csv
    saveDataInfoToCSV(dataInfo)

    # save class distribution to csv
    da.getClassDistribution().to_csv(
        f'{dirName}/info/class_distribution.csv')

    # save plots (to also draw them, pass draw=True as param)
    plotResults(trainScoresAcc, testScoresAcc, metric='acc', draw=False)
    plotResults(trainScoresBalancedAcc, testScoresBalancedAcc,
                metric='balanced_acc', draw=False)


if __name__ == '__main__':
    # MODE 1:
    # split data into train / test first (use the same 20% of ALL data as testing set for all training sets)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--size', type=float, nargs='+', default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument('-a','--alg', type=str, nargs='+', default=['logReg', 'svm', 'svc', 'dt', 'rf', 'ann'])
    args = parser.parse_args()

    # parameters
    selectedSizes = args.size
    selectedAlgorithms = args.alg
    randomSeeds = [42]
    PI = False
    SHOULD_RESAMPLE = False

    # create new directory for results of this run 
    # name of the folder can be passed as param (default name is timestamp)
    dirName = createDir()

    # save run settings
    settings = {
        'MODE': 1,
        'SELECTED_SIZES': selectedSizes,
        'SELECTED_ALGORITHMS': selectedAlgorithms,
        'PI_OPTIMIZED': PI,
        'RANDOM_SEED': randomSeeds[0],
        'RESAMPLED_DATASET': SHOULD_RESAMPLE
    }
    with open(f'{dirName}/info/run_settings.json', 'w') as fp:
        json.dump(settings, fp)

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
