from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_analysis import DataAnalysis
from data_preparation import DataPreparation


np.set_printoptions(precision=4)


def plotResults(trainScores, testScores, draw=False):
    # plot results
    sns.set()

    train = pd.DataFrame(data=trainScores)
    test = pd.DataFrame(data=testScores)

    # save to .csv
    train.to_csv(f'{dirName}/results/train_scores.csv')
    test.to_csv(f'{dirName}/results/test_scores.csv')

    trainResults = train.melt(
        'Samples', var_name='Algorithm', value_name='Accuracy')
    testResults = test.melt(
        'Samples', var_name='Algorithm', value_name='Accuracy')

    # plt.ticklabel_format(style='plain', axis='y')

    # training accuracy
    plt.close('all')
    plt.figure()
    trainingPlot = sns.pointplot(x='Samples', y='Accuracy', hue='Algorithm',
                                 data=trainResults, legend=True, legend_out=True)

    # testing accuracy
    plt.figure()
    testingPlot = sns.pointplot(x='Samples', y='Accuracy', hue='Algorithm',
                                data=testResults, legend=True, legend_out=True)

    # save plot images
    trainingPlot.get_figure().savefig(f'{dirName}/graphs/training_scores.png')
    testingPlot.get_figure().savefig(f'{dirName}/graphs/testing_scores.png')

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
        return newFolder
    except FileExistsError:
        print(f'Directory {name} already exists')
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

        # add accuracy / balanced_accuracy of each algorithm for plotting
        # train: 0 = accuracy, 2 = balanced_accuracy
        # test: 0 = accuracy, 1 = balanced_accuracy
        if algName in trainScores:
            trainScores[algName].append(train[2])
            testScores[algName].append(test[1])
        else:
            trainScores[algName] = [train[2]]
            testScores[algName] = [test[1]]


def main():
    if not dirName:
        return -1

    # use all available data
    sampleData = dp.returnData(1, randomSeed=42)
    saveDataset(sampleData, 'AD_dataset')

    # split data into X and y
    da = DataAnalysis(dirName=dirName)
    X, y = da.splitXY(sampleData)

    # split data into training (80%) and testing (20%) set
    xTrain, xTest, yTrain, yTest = da.splitTrainTest(
        X, y, trainSize=0.8, scale=True)
    saveDataset(xTrain.assign(normality=yTrain.values), 'AD_set_train')
    saveDataset(xTest.assign(normality=yTest.values), 'AD_set_test')

    # get data characteristics for current dataset size
    dataInfo[1] = da.getDataCharacteristics(yTrain, yTest)

    # Run predictions
    for datasetSize in selectedSizes:
        # get no. of samples and prediction accuracy
        noOfSamples, predictions = da.getScores(xTrain, xTest, yTrain, yTest, trainSize=datasetSize, randomSeeds=randomSeeds, selectedAlgorithms=selectedAlgorithms) if selectedAlgorithms else da.getScores(
            xTrain, yTrain, xTest, yTest, trainSize=datasetSize, randomSeeds=randomSeeds)

        # save results for graphs and .csv files
        savePredictionScores(noOfSamples, predictions, datasetSize)

    # save scores of all used algorithms to .csv
    saveScoresToCSV(fullTrain, fullTest)

    # save data info to .csv
    saveDataInfoToCSV(dataInfo)

    # save plots (to also draw them, pass draw=True as param)
    plotResults(trainScores, testScores, draw=False)


if __name__ == '__main__':
    # MODE 1:
    # split data into train / test first (use the same 20% of ALL data as test set for all training sets)

    # select dataset sizes (up to 1.0) and algorithms (all options: 'logReg', 'svm', 'dt', 'rf', 'ann')
    selectedSizes = [0.01, 0.05, 0.1, 0.15, 0.2]
    # selectedSizes = [0.2, 0.4, 0.6, 0.8, 1]
    selectedAlgorithms = ['logReg', 'svm', 'dt', 'rf', 'ann']
    # selectedAlgorithms = ['logReg', 'svm', 'dt', 'rf']

    # set number of repetitions and their respective random generator seeds
    randomSeeds = [42]

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
