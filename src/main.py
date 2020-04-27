from data_analysis import DataAnalysis
from data_preparation import DataPreparation

from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plotResults(trainScores, testScores, draw=False):
    # plot results
    sns.set()

    train = pd.DataFrame(data=trainScores)
    test = pd.DataFrame(data=testScores)

    trainResults = train.melt(
        'Samples', var_name='Algorithm', value_name='Accuracy')
    testResults = test.melt(
        'Samples', var_name='Algorithm', value_name='Accuracy')

    plt.ticklabel_format(style='plain', axis='y')

    # training accuracy
    plt.figure(1)
    trainingPlot = sns.pointplot(x='Samples', y='Accuracy', hue='Algorithm',
                                 data=trainResults, legend=True, legend_out=True).set_title('Training set')

    # testing accuracy
    plt.figure(2)
    testingPlot = sns.pointplot(x='Samples', y='Accuracy', hue='Algorithm',
                                data=testResults, legend=True, legend_out=True).set_title('Testing set')

    # save plot images
    trainingPlot.get_figure().savefig(dirName + '/training.png')
    testingPlot.get_figure().savefig(dirName + '/testing.png')

    plt.ticklabel_format(style='plain', axis='y')

    # draw
    if draw:
        plt.show()


def createDir(name=datetime.now().strftime("%d-%m-%Y (%H-%M-%S)")):
    resFolderName = 'results'
    try:
        # check if results folder exists already - if not, create it
        if not os.path.exists('../' + resFolderName):
            os.mkdir('../' + resFolderName)

        # create new folder for results from this run
        os.mkdir('../' + resFolderName + '/' + name)
        return '../' + resFolderName + '/' + name
    except FileExistsError:
        print("Directory ", name,  " already exists")
        return None


def saveData(data, fileName, folder='datasets'):
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

        fullTrainDF.to_csv(dirName + '/train_results_' + alg + '.csv')
        fullTestDF.to_csv(dirName + '/test_results_' + alg + '.csv')


def saveDataInfoToCSV(dataInfo):
    # map class indexes back to their labels
    normalityMapping = dp.getNormalityMapping()
    normalityClasses = [normalityMapping[index] for index in normalityMapping]
    # last row is sum of all classes
    normalityClasses.append('sum')
    dataInfoDF = pd.DataFrame(data=dataInfo, index=normalityClasses)
    # save to csv
    dataInfoDF.to_csv(dirName + '/data_info.csv')


def savePredictionScores(noOfSamples, predictions, datasetSize):
    # add no. of samples to dict for plotting
    trainScores['Samples'].append(noOfSamples)
    testScores['Samples'].append(noOfSamples)

    for algName in predictions:
        (train, test) = predictions[algName]

        # add all scores for .csv
        if algName not in fullTrain:
            fullTrain[algName] = {}
            fullTest[algName] = {}

        fullTrain[algName][datasetSize] = train
        fullTest[algName][datasetSize] = test

        # add accuracy of each algorithm for plotting
        if algName in trainScores:
            trainScores[algName].append(train[0])
            testScores[algName].append(test[0])
        else:
            trainScores[algName] = [train[0]]
            testScores[algName] = [test[0]]


def normalMode():
    da = DataAnalysis(dirName=dirName)

    # Run predictions
    for datasetSize in (selectedSizes or [0.2, 0.4, 0.6, 0.8, 1]):
        # get the percentage of all data
        sampleData = dp.returnData(datasetSize, randomSeed=10)
        saveData(sampleData, 'AD_dataset')

        # split data into X and y
        X, y = da.splitXY(sampleData)

        # split data into training (80%) and testing (20%) set
        xTrain, xTest, yTrain, yTest = da.splitTrainTest(X, y, trainSize=0.8)

        # get data characteristics for current dataset size
        dataInfo[datasetSize] = da.getDataCharacteristics(yTrain, yTest)

        # get no. of samples and prediction accuracy (trainSize is set to 1, so no further data splitting is done)
        # multiple runs make no sense here, since we always take whole set
        noOfSamples, predictions = da.getScores(xTrain, xTest, yTrain, yTest, trainSize=1, randomSeeds=[10], selectedAlgorithms=selectedAlgorithms) if selectedAlgorithms else da.getScores(
            xTrain, yTrain, xTest, yTest, trainSize=datasetSize, randomSeeds=randomSeeds)

        # save results for graphs and .csv files
        savePredictionScores(noOfSamples, predictions, datasetSize)


def splitDataFirstMode():
    # use all available data
    sampleData = dp.returnData(1, randomSeed=10)
    saveData(sampleData, 'AD_dataset')

    # split data into X and y
    da = DataAnalysis(dirName=dirName)
    X, y = da.splitXY(sampleData)

    # split data into training (80%) and testing (20%) set
    xTrain, xTest, yTrain, yTest = da.splitTrainTest(X, y, trainSize=0.8)
    saveData(xTrain.assign(normality=yTrain.values), 'AD_set_train')
    saveData(xTest.assign(normality=yTest.values), 'AD_set_test')

    # get data characteristics for current dataset size
    dataInfo[1] = da.getDataCharacteristics(yTrain, yTest)

    # Run predictions
    for datasetSize in (selectedSizes or [0.2, 0.4, 0.6, 0.8, 1]):
        # get no. of samples and prediction accuracy
        noOfSamples, predictions = da.getScores(xTrain, xTest, yTrain, yTest, trainSize=datasetSize, randomSeeds=randomSeeds, selectedAlgorithms=selectedAlgorithms) if selectedAlgorithms else da.getScores(
            xTrain, yTrain, xTest, yTest, trainSize=datasetSize, randomSeeds=randomSeeds)

        # save results for graphs and .csv files
        savePredictionScores(noOfSamples, predictions, datasetSize)


def main():
    if not dirName:
        return -1

    # run the predictions in the selected mode
    if mode == 0:
        normalMode()
    elif mode == 1:
        splitDataFirstMode()

    # save scores of all used algorithms to .csv
    saveScoresToCSV(fullTrain, fullTest)

    # save data info to .csv
    saveDataInfoToCSV(dataInfo)

    # save plots (to also draw them, pass draw=True as param)
    plotResults(trainScores, testScores, draw=False)


if __name__ == '__main__':
    # MODE:
    # 0 = take percentage of all data, then split it into train (80%) / test (20%)
    # 1 = split data into train / test first (use the same 20% of ALL data as test set for all training sets)
    mode = 1

    # select dataset sizes and algorithms (all options: 'logReg', 'svm', 'dt', 'rf', 'ann')
    selectedSizes = [0.2, 0.4, 0.6]
    selectedAlgorithms = ['dt', 'svm']

    # set number of repetitions and their respective random generator seeds
    randomSeeds = [20]

    # create new directory for results of this run
    # name of the folder can be passed as param (default name is timestamp)
    dirName = createDir()

    # init dicts to hold data
    dataInfo = {}
    fullTrain = {}
    fullTest = {}
    trainScores = {'Samples': []}    # scores for plotting
    testScores = {'Samples': []}     # scores for plotting

    # clean and preprocess data
    dp = DataPreparation('../data/mainSimulationAccessTraces.csv')
    dp.prepareData()

    # run predictions
    main()
