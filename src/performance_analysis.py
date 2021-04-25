import time
from datetime import datetime
import os
import json

from data_preparation import DataPreparation
from data_analysis import DataAnalysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV

from algorithms import Algorithms


class PerformanceAnalysis:
    def __init__(self, resultsDir=datetime.now().strftime("%d-%m-%Y(%H-%M-%S)"), verbose=0):
        self.predictions = None
        self.verbose = verbose
        self.dirName = 'results/performance_analysis'
        self.resultsDirName = f'{self.dirName}/{resultsDir}'
        self.model = {}

        if not os.path.exists(self.dirName):
            os.mkdir(self.dirName)

        if not os.path.exists(self.resultsDirName):
            os.mkdir(self.resultsDirName)

    def measureFitTime(self, alg, X, y, repeats=5):
        times = []
        calibratedModel = None

        for i in range(repeats):
            calibratedModel = None
            # Calibrated Classifier uses 5-fold CV by default
            calibratedModel = CalibratedClassifierCV(
                base_estimator=da.algorithms[alg][1])

            # fit the model
            startTime = time.time()
            calibratedModel.fit(X, y)
            duration = time.time() - startTime
            print(f'{alg} train, run {i+1}: {duration}s')
            times.append(duration)

        # Keep last trained model for predictions
        self.model[alg] = calibratedModel

        return times

    def measurePredictTime(self, alg, X, y, repeats=5):
        times = []

        for i in range(repeats):
            startTime = time.time()
            _ = self.model[alg].predict(X)
            duration = time.time() - startTime
            print(f'{alg} predict, run {i+1}: {duration}s')
            times.append(duration)

        return times

    def readFile(self, path):
        data = pd.read_csv(path, index_col=0)
        X = data.drop('normality', axis=1)
        y = data['normality']
        return X, y

    def savePlot(self, data, fileName):
        sns.set()
        plt.figure()
        plot = sns.barplot(data=data)
        plot.set(xlabel='Algorithms', ylabel='Time [s]')
        plot.get_figure().savefig(f'{self.resultsDirName}/{fileName}.png')

    def saveDataToCSV(self, data, fileName):
        data.to_csv(f'{self.resultsDirName}/{fileName}.csv')


if __name__ == '__main__':
    pa = PerformanceAnalysis()

    # variables
    SHOULD_RESAMPLE = True
    RANDOM_SEED = 42
    PI = True
    algs = ['logReg', 'svm', 'dt', 'rf', 'ann']
    # algs = ['dt']
    datasetSizes = [1]

    # save run settings
    settings = {
        'TRAINING_SET_SIZES': datasetSizes,
        'SELECTED_ALGORITHMS': algs,
        'PI_OPTIMIZED': PI,
        'RANDOM_SEED': RANDOM_SEED,
        'RESAMPLED_DATASET': SHOULD_RESAMPLE
    }
    with open(f'{pa.resultsDirName}/run_settings.json', 'w') as fp:
        json.dump(settings, fp)

    # clean and preprocess data
    print('Preparing data...')
    dp = DataPreparation('data/mainSimulationAccessTraces.csv')
    dp.prepareData()
    sampleData = dp.returnData(1, randomSeed=RANDOM_SEED)

    # split data into X and y
    da = DataAnalysis(dirName=None, mode=1, pi=PI)
    X, y = da.splitXY(sampleData)

    # split data into training (80%) and testing (20%) set
    print('Creating training and testing dataset...')
    xTrain, xTest, yTrain, yTest = da.splitTrainTest(
        X, y, trainSize=0.8, scale=True, resample=SHOULD_RESAMPLE, randomSeed=RANDOM_SEED)

    # For charts
    averageFitTimes = {}
    averagePredictTimes = {}
    setSizes = {}

    for alg in algs:
        algTag = da.algorithms[alg][0]

        averageFitTimes[alg] = {}
        averagePredictTimes[alg] = {}
        fitTimes = {}
        predictTimes = {}

        for size in datasetSizes:
            # split training set further into smaller sets]
            xTrainSmall, _, yTrainSmall, _ = da.splitTrainTest(
                xTrain, yTrain, trainSize=size, scale=False, resample=False, randomSeed=RANDOM_SEED)

            # save training/testing set size to file
            if size not in setSizes:
                setSizes[size] = {
                    'training': xTrainSmall.shape[0],
                    'testing': xTest.shape[0]
                }

            # measure train time
            measuredFitTimes = pa.measureFitTime(alg, xTrainSmall, yTrainSmall)
            fitTimes[size] = measuredFitTimes
            averageFitTimes[alg][size] = np.average(measuredFitTimes)

            # measure predict time
            measuredPredictTimes = pa.measurePredictTime(
                alg, xTest, yTest)
            predictTimes[size] = measuredPredictTimes
            averagePredictTimes[alg][size] = np.average(measuredPredictTimes)

        # Save alg results to a file
        pa.saveDataToCSV(pd.DataFrame(fitTimes), f'fit_times_{algTag}')
        pa.saveDataToCSV(pd.DataFrame(predictTimes), f'predict_times_{algTag}')

    # save times, set sizes to .csv
    pa.saveDataToCSV(pd.DataFrame(averageFitTimes), 'avg_fit_times')
    pa.saveDataToCSV(pd.DataFrame(averagePredictTimes), 'avg_predict_times')
    pa.saveDataToCSV(pd.DataFrame(setSizes), 'dataset_sizes')

