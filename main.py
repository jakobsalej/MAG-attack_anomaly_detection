from data_analysis import DataAnalysis
from data_preparation import DataPreparation

from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plotResults(train, test, draw=False):
    # plot results
    sns.set()

    trainResults = train.melt(
        'Samples', var_name='Algorithm', value_name='Accuracy')
    testResults = test.melt(
        'Samples', var_name='Algorithm', value_name='Accuracy')

    # training accuracy
    plt.figure(1)
    trainingPlot = sns.pointplot(x='Samples', y='Accuracy', hue='Algorithm',
                                 data=trainResults, legend=True, legend_out=True).set_title('Training set (avg. accuracy on 5-fold CV)')

    # testing accuracy
    plt.figure(2)
    testingPlot = sns.pointplot(x='Samples', y='Accuracy', hue='Algorithm',
                                data=testResults, legend=True, legend_out=True).set_title('Testing set')

    # save plot images
    trainingPlot.get_figure().savefig(dirName + '/training.png')
    testingPlot.get_figure().savefig(dirName + '/testing.png')

    # draw
    if draw:
        plt.show()


def createDir(name=datetime.now().strftime("%d-%m-%Y (%H:%M:%S)")):
    resFolderName = 'results'
    try:
        # check if results folder exists already - if not, create it
        if not os.path.exists(resFolderName):
            os.mkdir(resFolderName)

        # create new folder for results from this run
        os.mkdir(resFolderName + '/' + name)
        return resFolderName + '/' + name
    except FileExistsError:
        print("Directory ", name,  " already exists")
        return None


if __name__ == '__main__':
    # select dataset sizes and algorithms (all options: 'logReg', 'svm', 'dt', 'rf', 'ann')
    # set both to None to use default values
    selectedSizes = [0.2, 0.4, 0.6, 0.8, 1]
    selectedAlgorithms = ['logReg', 'svm', 'dt', 'rf']

    # create new directory for results of this run
    # name of the folder can be passed as param (default name is timestamp)
    dirName = createDir('all_but_ANN')

    # analyze and preprocess data
    dp = DataPreparation('data/mainSimulationAccessTraces.csv')
    dp.prepareData()

    # init dicts to hold data
    dataInfo = {}
    fullTrain = {}
    fullTest = {}
    trainScores = {'Samples': []}    # scores for plotting
    testScores = {'Samples': []}     # scores for plotting

    # Run predictions
    for datasetSize in (selectedSizes or [0.2, 0.4, 0.6, 0.8, 1]):
        sampleData = dp.returnData(datasetSize)
        da = DataAnalysis(sampleData)

        # get data characteristics for current dataset size
        dataInfo[datasetSize] = da.getDataCharacteristics()

        noOfSamples, predictions, fullTrainScores, fullTestScores = da.getScores(
            selectedAlgorithms) if selectedAlgorithms else da.getScores()

        # add scores
        fullTrain[datasetSize] = fullTrainScores
        fullTest[datasetSize] = fullTestScores

        # add no. of samples to dict for plotting
        trainScores['Samples'].append(noOfSamples)
        testScores['Samples'].append(noOfSamples)

        # add accuracies for each algorithm for plotting
        for algName in predictions:
            if algName in trainScores:
                trainScores[algName].append(predictions[algName][0])
                testScores[algName].append(predictions[algName][1])
            else:
                trainScores[algName] = [predictions[algName][0]]
                testScores[algName] = [predictions[algName][1]]

    # create pandas dataframes
    trainDF = pd.DataFrame(data=trainScores)
    testDF = pd.DataFrame(data=testScores)
    fullTrainDF = pd.DataFrame(data=fullTrain, index=[
                               'accuracy', 'std', 'balanced_accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'])
    fullTestDF = pd.DataFrame(data=fullTest, index=[
                              'accuracy', 'balanced_accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'])

    # data info where we map class indexes back to their labels
    normalityMapping = dp.getNormalityMapping()
    normalityClasses = [normalityMapping[index] for index in normalityMapping]
    # last row is sum of all classes
    rowLabels = normalityClasses.append('sum')
    dataInfoDF = pd.DataFrame(data=dataInfo, index=normalityClasses)

    # save to csv
    dataInfoDF.to_csv(dirName + '/data_info.csv')
    fullTrainDF.to_csv(dirName + '/train_results.csv')
    fullTestDF.to_csv(dirName + '/test_results.csv')

    # save plots (to also draw them, pass draw=True as param)
    plotResults(trainDF, testDF, draw=False)
