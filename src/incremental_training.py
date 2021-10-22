from datetime import datetime
import os
import json
from collections import Counter

from data_preparation import DataPreparation
from data_analysis import DataAnalysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score



class IncrementalTraining:
    def __init__(self, resultsDir=datetime.now().strftime("%d-%m-%Y(%H-%M-%S)"), verbose=0):
        self.predictions = None
        self.verbose = verbose
        self.dirName = 'results/incremental_training'
        self.resultsDirName = f'{self.dirName}/{resultsDir}'

        if not os.path.exists(self.dirName):
            os.mkdir(self.dirName)

        if not os.path.exists(self.resultsDirName):
            os.mkdir(self.resultsDirName)

        if not os.path.exists(f'{self.resultsDirName}/results'):
            os.mkdir(f'{self.resultsDirName}/results')

        if not os.path.exists(f'{self.resultsDirName}/graphs'):
            os.mkdir(f'{self.resultsDirName}/graphs')

    def saveDataToCSV(self, data, fileName):
        data.to_csv(f'{self.resultsDirName}/{fileName}.csv')


if __name__ == '__main__':
    pa = IncrementalTraining()

    # variables
    SHOULD_RESAMPLE = False
    RANDOM_SEED = 42
    BATCH_SIZE = 10000

    # save run settings
    settings = {
        'RANDOM_SEED': RANDOM_SEED,
        'RESAMPLED_DATASET': SHOULD_RESAMPLE,
        'BATCH_SIZE': BATCH_SIZE
    }
    with open(f'{pa.resultsDirName}/run_settings.json', 'w') as fp:
        json.dump(settings, fp)

    # clean and preprocess data
    print('Preparing data...')
    dp = DataPreparation('data/mainSimulationAccessTraces.csv')
    dp.prepareData()
    sampleData = dp.returnData(1, randomSeed=RANDOM_SEED)

    # split data into X and y
    da = DataAnalysis(dirName=pa.resultsDirName, mode=2, pi=False)
    X, y = da.splitXY(sampleData)
    allClasses = np.unique(y)

    # split data into training (80%) and testing (20%) set
    print('Creating training and testing dataset...')
    xTrain, xTest, yTrain, yTest = da.splitTrainTest(
        X, y, trainSize=0.8, scale=True, resample=SHOULD_RESAMPLE, randomSeed=RANDOM_SEED)

    trainingAcc = {}
    testingAcc = {}
    trainingBalancedAcc = {}
    testingBalancedAcc = {}

    sgd = SGDClassifier(n_jobs=-1)

    noOfSamples, _ = xTrain.shape
    batchNumber = 0
    startIndex = 0
    while startIndex < noOfSamples:
        print('Batch number', batchNumber)
        endIndex = min(startIndex + BATCH_SIZE, noOfSamples)
        
        # current batch
        xTrainBatch = xTrain[startIndex:endIndex]
        yTrainBatch = yTrain[startIndex:endIndex]

        # get class distribution for current batch
        batchName = f'batch_{batchNumber + 1}'
        da.saveClassDistribution(batchName, yTrainBatch)

        # train
        sgd.partial_fit(xTrainBatch, yTrainBatch, allClasses)

        # test training
        predictedBatch = sgd.predict(xTrainBatch)
        trainingAcc[batchNumber] = accuracy_score(yTrainBatch, predictedBatch)
        trainingBalancedAcc[batchNumber] = balanced_accuracy_score(yTrainBatch, predictedBatch)
        print('Train Balanced acc', trainingBalancedAcc[batchNumber])


        # test testing
        predictedTest = sgd.predict(xTest)
        testingAcc[batchNumber] = accuracy_score(yTest, predictedTest)
        testingBalancedAcc[batchNumber] = balanced_accuracy_score(yTest, predictedTest)
        print('Test Balanced acc', testingBalancedAcc[batchNumber])

        # save test set CM
        da.saveConfusionMatrix(yTest, predictedTest, batchName)
        
        startIndex = endIndex
        batchNumber += 1

        # if (endIndex == noOfSamples - 1):
        #     break

    # save class dists to .csv
    da.getClassDistribution().to_csv(f'{pa.resultsDirName}/class_distribution.csv')

    # save scores to .csv
    print(trainingAcc)
    pa.saveDataToCSV(pd.DataFrame(trainingAcc, index=[0]), 'training_accuracy')
    pa.saveDataToCSV(pd.DataFrame(trainingBalancedAcc, index=[0]), 'training_balanced_accuracy')
    pa.saveDataToCSV(pd.DataFrame(testingAcc, index=[0]), 'testing_accuracy')
    pa.saveDataToCSV(pd.DataFrame(testingBalancedAcc, index=[0]), 'testing_balanced_accuracy')

